/**
 * LanceDatabase - Multi-table SQL database
 */

class LanceDatabase {
    constructor() {
        this.tables = new Map(); // name -> RemoteLanceDataset
        this.aliases = new Map(); // alias -> table name
        // Query plan cache
        this._planCache = new Map(); // normalized SQL -> { plan, hits, lastUsed }
        this._planCacheMaxSize = 100;
        // In-memory tables (ephemeral)
        this.memoryTables = new Map(); // name -> MemoryTable
    }

    /**
     * Register a table with a name
     * @param {string} name - Table name
     * @param {RemoteLanceDataset} dataset - Dataset instance
     */
    register(name, dataset) {
        this.tables.set(name, dataset);
    }

    /**
     * Register a remote dataset by URL
     * @param {string} name - Table name
     * @param {string} url - Dataset URL
     * @param {Object} options - Dataset options (version, etc.)
     */
    async registerRemote(name, url, options = {}) {
        // Assume LanceQL is globally available or passed as parameter
        const lanceql = window.lanceql || globalThis.lanceql;
        if (!lanceql) {
            throw new Error('LanceQL WASM module not loaded. Call LanceQL.load() first.');
        }

        const dataset = await lanceql.openDataset(url, options);
        this.register(name, dataset);
        return dataset;
    }

    /**
     * Get a table by name or alias
     */
    getTable(name) {
        // Check aliases first
        const actualName = this.aliases.get(name) || name;
        const table = this.tables.get(actualName);
        if (!table) {
            throw new Error(`Table '${name}' not found. Did you forget to register it?`);
        }
        return table;
    }

    /**
     * Execute SQL query (supports SELECT with JOINs, CTEs, SET operations, and EXPLAIN)
     */
    async executeSQL(sql) {
        // Check plan cache first
        const cachedPlan = this._getCachedPlan(sql);
        let ast;

        if (cachedPlan) {
            ast = cachedPlan;
        } else {
            // Parse SQL
            const lexer = new SQLLexer(sql);
            const tokens = lexer.tokenize();
            const parser = new SQLParser(tokens);
            ast = parser.parse();

            // Cache the plan (unless it's EXPLAIN which is meta)
            if (ast.type !== 'EXPLAIN') {
                this._setCachedPlan(sql, ast);
            }
        }

        // Handle EXPLAIN - return plan without executing
        if (ast.type === 'EXPLAIN') {
            return this._explainQuery(ast.statement);
        }

        // Handle memory table operations (CREATE, DROP, INSERT, UPDATE, DELETE)
        if (ast.type === 'CREATE_TABLE') {
            return this._executeCreateTable(ast);
        }
        if (ast.type === 'DROP_TABLE') {
            return this._executeDropTable(ast);
        }
        if (ast.type === 'INSERT') {
            return this._executeInsert(ast);
        }
        if (ast.type === 'UPDATE') {
            return this._executeUpdate(ast);
        }
        if (ast.type === 'DELETE') {
            return this._executeDelete(ast);
        }

        if (ast.type === 'SET_OPERATION') {
            return this._executeSetOperation(ast);
        }

        if (ast.type !== 'SELECT') {
            throw new Error('Only SELECT queries are supported in LanceDatabase');
        }

        // Handle CTEs - materialize them first
        if (ast.ctes && ast.ctes.length > 0) {
            return this._executeWithCTEs(ast);
        }

        // No joins - simple single-table query
        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }

        // Multi-table query with JOINs
        return this._executeJoin(ast);
    }

    /**
     * Execute query with CTEs
     */
    async _executeWithCTEs(ast) {
        // Create a temporary executor for CTE materialization
        const cteExecutor = new SQLExecutor({ columnNames: [] });
        cteExecutor.setDatabase(this);
        await cteExecutor.materializeCTEs(ast.ctes, this);

        // Check if main query references a CTE
        const mainTableName = ast.from?.name?.toLowerCase() || ast.from?.table?.toLowerCase();
        if (mainTableName && cteExecutor._cteResults.has(mainTableName)) {
            // Execute main query against CTE result
            return cteExecutor._executeOnInMemoryData(ast, cteExecutor._cteResults.get(mainTableName));
        }

        // Otherwise execute against actual tables
        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }
        return this._executeJoin(ast);
    }

    /**
     * Execute SET operation (UNION, INTERSECT, EXCEPT)
     */
    async _executeSetOperation(ast) {
        // Execute left and right sides
        const leftResult = await this.executeSQL(this._astToSQL(ast.left));
        const rightResult = await this.executeSQL(this._astToSQL(ast.right));

        if (leftResult.columns.length !== rightResult.columns.length) {
            throw new Error('SET operations require same number of columns');
        }

        const rowKey = row => JSON.stringify(row);
        let combinedRows;

        switch (ast.operator) {
            case 'UNION':
                combinedRows = [...leftResult.rows, ...rightResult.rows];
                if (!ast.all) {
                    const seen = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seen.has(key)) return false;
                        seen.add(key);
                        return true;
                    });
                }
                break;

            case 'INTERSECT':
                const rightKeys = new Set(rightResult.rows.map(rowKey));
                combinedRows = leftResult.rows.filter(row => rightKeys.has(rowKey(row)));
                if (!ast.all) {
                    const seenI = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seenI.has(key)) return false;
                        seenI.add(key);
                        return true;
                    });
                }
                break;

            case 'EXCEPT':
                const excludeKeys = new Set(rightResult.rows.map(rowKey));
                combinedRows = leftResult.rows.filter(row => !excludeKeys.has(rowKey(row)));
                if (!ast.all) {
                    const seenE = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seenE.has(key)) return false;
                        seenE.add(key);
                        return true;
                    });
                }
                break;

            default:
                throw new Error(`Unknown SET operator: ${ast.operator}`);
        }

        // Apply ORDER BY to combined result
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            leftResult.columns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            combinedRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = ob.direction === 'DESC' ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET to combined result
        const offset = ast.offset || 0;
        if (offset > 0) combinedRows = combinedRows.slice(offset);
        if (ast.limit) combinedRows = combinedRows.slice(0, ast.limit);

        return { columns: leftResult.columns, rows: combinedRows, total: combinedRows.length };
    }

    /**
     * Convert AST back to SQL (for recursive SET operation execution)
     */
    _astToSQL(ast) {
        if (ast.type === 'SET_OPERATION') {
            const left = this._astToSQL(ast.left);
            const right = this._astToSQL(ast.right);
            const op = ast.operator + (ast.all ? ' ALL' : '');
            return `(${left}) ${op} (${right})`;
        }

        // Build SELECT statement
        let sql = ast.distinct ? 'SELECT DISTINCT ' : 'SELECT ';
        sql += ast.columns.map(col => {
            if (col.expr?.type === 'star') return '*';
            const expr = this._exprToSQL(col.expr);
            return col.alias ? `${expr} AS ${col.alias}` : expr;
        }).join(', ');

        if (ast.from) {
            const tableName = ast.from.name || ast.from.table;
            sql += ` FROM ${tableName}`;
            if (ast.from.alias) sql += ` AS ${ast.from.alias}`;
        }

        if (ast.joins) {
            for (const join of ast.joins) {
                const rightTable = join.table?.name || join.table?.table;
                sql += ` ${join.type} ${rightTable}`;
                if (join.alias) sql += ` AS ${join.alias}`;
                if (join.on) sql += ` ON ${this._exprToSQL(join.on)}`;
            }
        }

        if (ast.where) sql += ` WHERE ${this._exprToSQL(ast.where)}`;
        if (ast.groupBy?.length) sql += ` GROUP BY ${ast.groupBy.join(', ')}`;
        if (ast.having) sql += ` HAVING ${this._exprToSQL(ast.having)}`;
        if (ast.orderBy?.length) {
            sql += ` ORDER BY ${ast.orderBy.map(o => `${o.column} ${o.direction || 'ASC'}`).join(', ')}`;
        }
        if (ast.limit) sql += ` LIMIT ${ast.limit}`;
        if (ast.offset) sql += ` OFFSET ${ast.offset}`;

        return sql;
    }

    /**
     * Convert expression AST to SQL string
     */
    _exprToSQL(expr) {
        if (!expr) return '';
        switch (expr.type) {
            case 'literal':
                if (expr.value === null) return 'NULL';
                if (typeof expr.value === 'string') return `'${expr.value.replace(/'/g, "''")}'`;
                return String(expr.value);
            case 'column':
                return expr.table ? `${expr.table}.${expr.column}` : expr.column;
            case 'star':
                return '*';
            case 'binary':
                return `(${this._exprToSQL(expr.left)} ${expr.operator} ${this._exprToSQL(expr.right)})`;
            case 'unary':
                return `(${expr.operator} ${this._exprToSQL(expr.operand)})`;
            case 'call':
                const args = expr.args.map(a => this._exprToSQL(a)).join(', ');
                return `${expr.name}(${expr.distinct ? 'DISTINCT ' : ''}${args})`;
            case 'in':
                const vals = expr.values.map(v => this._exprToSQL(v)).join(', ');
                return `${this._exprToSQL(expr.expr)} IN (${vals})`;
            case 'between':
                return `${this._exprToSQL(expr.expr)} BETWEEN ${this._exprToSQL(expr.low)} AND ${this._exprToSQL(expr.high)}`;
            case 'like':
                return `${this._exprToSQL(expr.expr)} LIKE ${this._exprToSQL(expr.pattern)}`;
            default:
                return '';
        }
    }

    /**
     * Execute single-table query (no joins)
     */
    async _executeSingleTable(ast) {
        if (!ast.from) {
            throw new Error('FROM clause required');
        }

        // Get table
        let tableName = ast.from.name || ast.from.table;
        if (!tableName && ast.from.url) {
            throw new Error('Single-table queries must use registered table names, not URLs');
        }

        const tableNameLower = tableName.toLowerCase();

        // Check memory table first
        if (this.memoryTables.has(tableNameLower)) {
            const memTable = this.memoryTables.get(tableNameLower);
            const executor = new SQLExecutor({ columnNames: memTable.columns });
            return executor._executeOnInMemoryData(ast, memTable.toInMemoryData());
        }

        // Otherwise use remote dataset
        const dataset = this.getTable(tableName);

        // Build SQL and execute
        const executor = new SQLExecutor(dataset);
        return executor.execute(ast);
    }

    /**
     * Execute multi-table query with JOINs
     */
    async _executeJoin(ast) {
        console.log('[LanceDatabase] Executing JOIN query:', ast);

        // Extract table references
        const leftTableName = ast.from.name || ast.from.table;
        const leftAlias = ast.from.alias || leftTableName;

        // Register alias
        if (ast.from.alias) {
            this.aliases.set(ast.from.alias, leftTableName);
        }

        // Process joins iteratively: (A JOIN B) JOIN C
        // Each join's result becomes the left input for the next
        console.log(`[LanceDatabase] Processing ${ast.joins.length} JOIN(s)`);

        let currentResult = null;  // In-memory intermediate result
        let currentAlias = leftAlias;
        let currentTableName = leftTableName;
        let leftDataset = this.getTable(leftTableName);

        for (let i = 0; i < ast.joins.length; i++) {
            const join = ast.joins[i];
            const rightTableName = join.table.name || join.table.table;
            const rightAlias = join.alias || rightTableName;

            // Register right table alias
            if (join.alias) {
                this.aliases.set(join.alias, rightTableName);
            }

            console.log(`[LanceDatabase] JOIN ${i + 1}/${ast.joins.length}: ${currentTableName} (${currentAlias}) ${join.type} ${rightTableName} (${rightAlias})`);

            // Get right dataset
            const rightDataset = this.getTable(rightTableName);

            // Build AST for this single join
            const singleJoinAst = {
                ...ast,
                joins: [join],  // Only this join
                // For intermediate joins, don't apply final limit/projection
                limit: (i === ast.joins.length - 1) ? ast.limit : undefined,
                columns: (i === ast.joins.length - 1) ? ast.columns : [{ type: 'column', column: '*' }]
            };

            // Execute hash join
            if (currentResult === null) {
                // First join: left is a dataset
                currentResult = await this._hashJoin(
                    leftDataset,
                    rightDataset,
                    singleJoinAst,
                    { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
                );
            } else {
                // Subsequent joins: left is in-memory result
                currentResult = await this._hashJoinWithInMemoryLeft(
                    currentResult,
                    rightDataset,
                    singleJoinAst,
                    { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
                );
            }

            // Update current state for next iteration
            currentAlias = `${currentAlias}_${rightAlias}`;  // Compound alias for tracing
            currentTableName = `(${currentTableName} JOIN ${rightTableName})`;
        }

        return currentResult;
    }

    /**
     * Execute hash join between two datasets using OPFS for intermediate storage.
     * This enables TB-scale joins in the browser by spilling to disk instead of RAM.
     */
    async _hashJoin(leftDataset, rightDataset, ast, context) {
        const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
        const join = ast.joins[0];
        const joinType = join.type || 'INNER';

        // For CROSS JOIN, no ON condition is required
        const joinCondition = join.on;
        if (joinType !== 'CROSS') {
            if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
                throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
            }
        }

        // For CROSS JOIN, use simplified execution (no keys needed)
        let leftKey, rightKey, leftSQL, rightSQL;

        if (joinType === 'CROSS') {
            // CROSS JOIN: select all columns, no join keys
            leftKey = null;
            rightKey = null;
            leftSQL = `SELECT * FROM ${leftTableName}`;
            rightSQL = `SELECT * FROM ${rightTableName}`;
            console.log('[LanceDatabase] CROSS JOIN - no keys, cartesian product');
        } else {
            // Use QueryPlanner to generate optimized execution plan
            const planner = new QueryPlanner();
            const plan = planner.plan(ast, context);

            // Extract columns and keys from the plan
            leftKey = plan.join.leftKey;
            rightKey = plan.join.rightKey;
            const leftColumns = plan.leftScan.columns;
            const rightColumns = plan.rightScan.columns;
            const leftFilters = plan.leftScan.filters;
            const rightFilters = plan.rightScan.filters;

            // Build SQL queries for streaming
            const leftColsWithKey = leftColumns.includes('*')
                ? ['*']
                : [...new Set([leftKey, ...leftColumns])];

            let leftWhereClause = '';
            if (leftFilters.length > 0) {
                leftWhereClause = ` WHERE ${leftFilters.map(f => this._filterToSQL(f)).join(' AND ')}`;
            }
            leftSQL = `SELECT ${leftColsWithKey.join(', ')} FROM ${leftTableName}${leftWhereClause}`;

            const rightColsWithKey = rightColumns.includes('*')
                ? ['*']
                : [...new Set([rightKey, ...rightColumns])];

            let rightWhereClause = '';
            if (rightFilters.length > 0) {
                rightWhereClause = ` WHERE ${rightFilters.map(f => this._filterToSQL(f)).join(' AND ')}`;
            }
            rightSQL = `SELECT ${rightColsWithKey.join(', ')} FROM ${rightTableName}${rightWhereClause}`;
        }

        console.log('[LanceDatabase] OPFS-backed hash join starting...');
        console.log('[LanceDatabase] Left query:', leftSQL);

        // Initialize OPFS storage
        await opfsStorage.open();

        // Create OPFS join executor
        const joinExecutor = new OPFSJoinExecutor(opfsStorage);

        // Create streaming executors
        const leftExecutor = new SQLExecutor(leftDataset);
        const rightExecutor = new SQLExecutor(rightDataset);

        // Semi-join optimization: partition left first and collect keys
        const leftStream = leftExecutor.executeStream(leftSQL);
        const leftMeta = await joinExecutor._partitionToOPFS(leftStream, leftKey, 'left', true);
        console.log(`[LanceDatabase] Left partitioned: ${leftMeta.totalRows} rows, ${leftMeta.collectedKeys?.size || 0} unique keys`);

        // Build optimized right SQL with IN clause if we have reasonable key count
        let optimizedRightSQL = rightSQL;
        const maxKeysForInClause = 1000;  // Don't create huge IN clauses
        if (leftMeta.collectedKeys && leftMeta.collectedKeys.size > 0 &&
            leftMeta.collectedKeys.size <= maxKeysForInClause) {
            const inClause = this._buildInClause(rightKey, leftMeta.collectedKeys);
            optimizedRightSQL = this._appendWhereClause(rightSQL, inClause);
            console.log(`[LanceDatabase] Semi-join optimization: added IN clause with ${leftMeta.collectedKeys.size} keys`);
        }
        console.log('[LanceDatabase] Right query:', optimizedRightSQL);

        // Create right stream with optimized SQL
        const rightStream = rightExecutor.executeStream(optimizedRightSQL);

        // Execute OPFS-backed join with pre-partitioned left
        const results = [];
        let resultColumns = null;

        try {
            for await (const chunk of joinExecutor.executeHashJoin(
                null,  // leftStream already partitioned
                rightStream,
                leftKey,
                rightKey,
                {
                    limit: ast.limit || Infinity,
                    leftAlias,
                    rightAlias,
                    joinType: join.type || 'INNER',
                    prePartitionedLeft: leftMeta
                }
            )) {
                if (!resultColumns) {
                    resultColumns = chunk.columns;
                }
                results.push(...chunk.rows);

                // Early exit if we have enough rows
                if (ast.limit && results.length >= ast.limit) {
                    break;
                }
            }
        } catch (e) {
            console.error('[LanceDatabase] OPFS join failed:', e);
            throw e;
        }

        // Get stats
        const stats = joinExecutor.getStats();
        console.log('[LanceDatabase] OPFS Join Stats:', stats);

        // If no results, return empty
        if (!resultColumns || results.length === 0) {
            return { columns: [], rows: [], total: 0, opfsStats: stats };
        }

        // Apply projection
        const projectedResults = this._applyProjection(
            results,
            resultColumns,
            plan.projection,
            leftAlias,
            rightAlias
        );

        // Apply LIMIT
        const limitedResults = ast.limit
            ? projectedResults.rows.slice(0, ast.limit)
            : projectedResults.rows;

        return {
            columns: projectedResults.columns,
            rows: limitedResults,
            total: limitedResults.length,
            opfsStats: stats  // Include OPFS stats in result
        };
    }

    /**
     * Execute hash join with in-memory left side (for multiple JOINs).
     * The left side comes from a previous join's result.
     */
    async _hashJoinWithInMemoryLeft(leftResult, rightDataset, ast, context) {
        const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
        const join = ast.joins[0];
        const joinType = join.type || 'INNER';

        // For CROSS JOIN, no ON condition required
        const joinCondition = join.on;
        if (joinType !== 'CROSS') {
            if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
                throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
            }
        }

        // Extract join keys
        let leftKey, rightKey;
        if (joinType === 'CROSS') {
            leftKey = null;
            rightKey = null;
        } else {
            // Extract keys from ON condition
            const leftExpr = joinCondition.left;
            const rightExpr = joinCondition.right;

            // Determine which side refers to left vs right
            const leftCol = leftExpr.column;
            const rightCol = rightExpr.column;

            // Find which column is in the left result columns
            const leftColsSet = new Set(leftResult.columns.map(c => {
                const parts = c.split('.');
                return parts[parts.length - 1];  // Get base column name
            }));

            if (leftColsSet.has(leftCol)) {
                leftKey = leftCol;
                rightKey = rightCol;
            } else {
                leftKey = rightCol;
                rightKey = leftCol;
            }
        }

        console.log(`[LanceDatabase] Multi-JOIN: left in-memory (${leftResult.rows.length} rows), right: ${rightTableName}`);

        // Build right SQL - select all for now
        let rightSQL = `SELECT * FROM ${rightTableName}`;

        // Semi-join optimization: use in-memory left keys to filter right
        const maxKeysForInClause = 1000;
        if (leftKey && joinType !== 'CROSS') {
            const leftKeyIndex = this._findColumnIndex(leftResult.columns, leftKey);
            if (leftKeyIndex !== -1) {
                const leftKeys = new Set();
                for (const row of leftResult.rows) {
                    const key = row[leftKeyIndex];
                    if (key !== null && key !== undefined) {
                        leftKeys.add(key);
                    }
                }
                if (leftKeys.size > 0 && leftKeys.size <= maxKeysForInClause) {
                    const inClause = this._buildInClause(rightKey, leftKeys);
                    rightSQL = this._appendWhereClause(rightSQL, inClause);
                    console.log(`[LanceDatabase] Multi-JOIN semi-join: ${leftKeys.size} keys`);
                }
            }
        }

        // Execute right query
        const rightExecutor = new SQLExecutor(rightDataset);
        const rightResult = await rightExecutor.execute(new SQLParser(new SQLLexer(rightSQL).tokenize()).parse());

        // Find key indices for in-memory hash join
        const leftKeyIndex = leftKey ? this._findColumnIndex(leftResult.columns, leftKey) : -1;
        const rightKeyIndex = rightKey ? this._findColumnIndex(rightResult.columns, rightKey) : -1;

        // Build result columns
        const resultColumns = [
            ...leftResult.columns,
            ...rightResult.columns.map(c => `${rightAlias}.${c}`)
        ];

        // Execute in-memory hash join
        const results = [];
        const rightNulls = new Array(rightResult.columns.length).fill(null);
        const leftNulls = new Array(leftResult.columns.length).fill(null);

        if (joinType === 'CROSS') {
            // Cartesian product
            for (const leftRow of leftResult.rows) {
                for (const rightRow of rightResult.rows) {
                    results.push([...leftRow, ...rightRow]);
                    if (ast.limit && results.length >= ast.limit) break;
                }
                if (ast.limit && results.length >= ast.limit) break;
            }
        } else {
            // Build hash table from right side
            const rightHash = new Map();
            for (const row of rightResult.rows) {
                const key = row[rightKeyIndex];
                if (key !== null && key !== undefined) {
                    if (!rightHash.has(key)) rightHash.set(key, []);
                    rightHash.get(key).push(row);
                }
            }

            // Track matched right rows for FULL/RIGHT joins
            const matchedRightRows = (joinType === 'FULL' || joinType === 'RIGHT')
                ? new Set() : null;

            // Probe with left rows
            for (const leftRow of leftResult.rows) {
                const key = leftRow[leftKeyIndex];
                const rightMatches = rightHash.get(key) || [];

                if (rightMatches.length > 0) {
                    for (const rightRow of rightMatches) {
                        results.push([...leftRow, ...rightRow]);
                        if (matchedRightRows) {
                            matchedRightRows.add(rightResult.rows.indexOf(rightRow));
                        }
                        if (ast.limit && results.length >= ast.limit) break;
                    }
                } else if (joinType === 'LEFT' || joinType === 'FULL') {
                    // Left row with no match
                    results.push([...leftRow, ...rightNulls]);
                }
                if (ast.limit && results.length >= ast.limit) break;
            }

            // Add unmatched right rows for RIGHT/FULL joins
            if ((joinType === 'RIGHT' || joinType === 'FULL') && matchedRightRows) {
                for (let i = 0; i < rightResult.rows.length; i++) {
                    if (!matchedRightRows.has(i)) {
                        results.push([...leftNulls, ...rightResult.rows[i]]);
                        if (ast.limit && results.length >= ast.limit) break;
                    }
                }
            }
        }

        // Apply projection if this is the final join
        const limitedResults = ast.limit ? results.slice(0, ast.limit) : results;

        return {
            columns: resultColumns,
            rows: limitedResults,
            total: limitedResults.length
        };
    }

    /**
     * Find column index by name, handling qualified names (table.column)
     */
    _findColumnIndex(columns, columnName) {
        // Try exact match first
        let idx = columns.indexOf(columnName);
        if (idx !== -1) return idx;

        // Try with any table prefix (for qualified names like "users.id")
        for (let i = 0; i < columns.length; i++) {
            const col = columns[i];
            const parts = col.split('.');
            if (parts[parts.length - 1] === columnName) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Convert filter expression to SQL WHERE clause
     * Note: Strips table aliases since pushed-down queries are single-table
     */
    _filterToSQL(expr) {
        if (!expr) return '';

        if (expr.type === 'binary') {
            const left = this._filterToSQL(expr.left);
            const right = this._filterToSQL(expr.right);
            return `${left} ${expr.op} ${right}`;
        } else if (expr.type === 'column') {
            // Strip table alias - pushed query is single-table
            return expr.column;
        } else if (expr.type === 'literal') {
            if (typeof expr.value === 'string') {
                // Escape single quotes to prevent SQL injection
                const escaped = expr.value.replace(/'/g, "''");
                return `'${escaped}'`;
            }
            if (expr.value === null) return 'NULL';
            return String(expr.value);
        } else if (expr.type === 'call') {
            const args = (expr.args || []).map(a => this._filterToSQL(a)).join(', ');
            return `${expr.name}(${args})`;
        } else if (expr.type === 'in') {
            const col = this._filterToSQL(expr.expr);
            const vals = expr.values.map(v => this._filterToSQL(v)).join(', ');
            return `${col} IN (${vals})`;
        } else if (expr.type === 'between') {
            const col = this._filterToSQL(expr.expr);
            const low = this._filterToSQL(expr.low);
            const high = this._filterToSQL(expr.high);
            return `${col} BETWEEN ${low} AND ${high}`;
        } else if (expr.type === 'like') {
            const col = this._filterToSQL(expr.expr);
            const pattern = this._filterToSQL(expr.pattern);
            return `${col} LIKE ${pattern}`;
        } else if (expr.type === 'unary') {
            const operand = this._filterToSQL(expr.operand);
            if (expr.op === 'NOT') return `NOT ${operand}`;
            return `${expr.op}${operand}`;
        }

        console.warn('[LanceDB] Unknown filter expression type:', expr.type);
        return '';
    }

    /**
     * Apply projection to select only requested columns from joined result
     */
    _applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
        // Handle SELECT *
        if (projection.includes('*')) {
            return { columns: allColumns, rows };
        }

        // Build column mapping
        const projectedColumns = [];
        const columnIndices = [];

        for (const col of projection) {
            if (col === '*') {
                // Already handled above
                continue;
            }

            let idx = -1;
            let outputColName = col.column;

            if (col.table) {
                // Try exact match with table prefix first (most specific)
                const qualifiedName = `${col.table}.${col.column}`;
                idx = allColumns.indexOf(qualifiedName);
                outputColName = qualifiedName;
            }

            if (idx === -1) {
                // Fallback: find first column ending with this column name
                idx = allColumns.findIndex(c => c === col.column || c.endsWith(`.${col.column}`));
                if (idx !== -1) {
                    outputColName = allColumns[idx];
                }
            }

            if (idx !== -1) {
                projectedColumns.push(col.alias || outputColName);
                columnIndices.push(idx);
            }
        }

        // Apply projection
        const projectedRows = rows.map(row =>
            columnIndices.map(idx => row[idx])
        );

        return { columns: projectedColumns, rows: projectedRows };
    }

    /**
     * Extract column name from expression
     */
    _extractColumnFromExpr(expr, expectedTable) {
        if (expr.type === 'column') {
            // Handle table.column syntax
            if (expr.table && expr.table !== expectedTable) {
                // Column belongs to different table
                return null;
            }
            return expr.column;
        }
        throw new Error(`Invalid join condition expression: ${JSON.stringify(expr)}`);
    }

    /**
     * Get columns needed for a specific table from SELECT list
     */
    _getColumnsForTable(selectColumns, tableAlias) {
        const columns = [];

        for (const item of selectColumns) {
            if (item.type === 'star') {
                // SELECT * - need to fetch all columns (TODO: get schema)
                return ['*'];
            }

            if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                if (!col.table || col.table === tableAlias) {
                    columns.push(col.column);
                }
            }
        }

        return columns.length > 0 ? columns : ['*'];
    }

    /**
     * Build an IN clause for semi-join optimization
     * @param {string} column - Column name for IN clause
     * @param {Set} keys - Unique key values collected from left table
     * @returns {string} SQL IN clause fragment
     */
    _buildInClause(column, keys) {
        const values = Array.from(keys).map(k => {
            if (typeof k === 'string') {
                return `'${k.replace(/'/g, "''")}'`;  // Escape single quotes
            }
            if (k === null) return 'NULL';
            return String(k);
        }).join(', ');
        return `${column} IN (${values})`;
    }

    /**
     * Append a WHERE clause or AND condition to existing SQL
     * @param {string} sql - Existing SQL query
     * @param {string} clause - Condition to add
     * @returns {string} SQL with added condition
     */
    _appendWhereClause(sql, clause) {
        const upperSQL = sql.toUpperCase();
        if (upperSQL.includes('WHERE')) {
            // Insert after WHERE keyword
            return sql.replace(/WHERE\s+/i, `WHERE ${clause} AND `);
        }
        // Find FROM table (with optional alias) and add WHERE after
        // Match: FROM tablename or FROM tablename alias
        return sql.replace(/FROM\s+(\w+)(\s+\w+)?/i, (match) => `${match} WHERE ${clause}`);
    }

    // ========================================================================
    // Phase 9: Query Optimization Methods
    // ========================================================================

    /**
     * Get cached query plan
     * @param {string} sql - SQL query string
     * @returns {Object|null} Cached plan or null
     */
    _getCachedPlan(sql) {
        const normalized = this._normalizeSQL(sql);
        const cached = this._planCache.get(normalized);
        if (cached) {
            cached.hits++;
            cached.lastUsed = Date.now();
            return cached.plan;
        }
        return null;
    }

    /**
     * Cache a query plan
     * @param {string} sql - SQL query string
     * @param {Object} plan - Parsed AST plan
     */
    _setCachedPlan(sql, plan) {
        const normalized = this._normalizeSQL(sql);

        // LRU eviction if at capacity
        if (this._planCache.size >= this._planCacheMaxSize) {
            let oldest = null;
            let oldestTime = Infinity;
            for (const [key, value] of this._planCache) {
                if (value.lastUsed < oldestTime) {
                    oldestTime = value.lastUsed;
                    oldest = key;
                }
            }
            if (oldest) this._planCache.delete(oldest);
        }

        this._planCache.set(normalized, {
            plan,
            hits: 0,
            lastUsed: Date.now(),
            created: Date.now()
        });
    }

    /**
     * Normalize SQL for cache key (remove extra whitespace, lowercase)
     * @param {string} sql - SQL query string
     * @returns {string} Normalized SQL
     */
    _normalizeSQL(sql) {
        return sql.trim().replace(/\s+/g, ' ').toLowerCase();
    }

    /**
     * Clear the query plan cache
     */
    clearPlanCache() {
        this._planCache.clear();
    }

    /**
     * Get plan cache statistics
     * @returns {Object} Cache stats
     */
    getPlanCacheStats() {
        let totalHits = 0;
        for (const v of this._planCache.values()) {
            totalHits += v.hits;
        }
        return {
            size: this._planCache.size,
            maxSize: this._planCacheMaxSize,
            totalHits
        };
    }

    /**
     * Optimize expression with constant folding and boolean simplification
     * @param {Object} expr - Expression AST node
     * @returns {Object} Optimized expression
     */
    _optimizeExpr(expr) {
        if (!expr) return expr;

        // Recursively optimize children
        if (expr.left) expr.left = this._optimizeExpr(expr.left);
        if (expr.right) expr.right = this._optimizeExpr(expr.right);
        if (expr.operand) expr.operand = this._optimizeExpr(expr.operand);
        if (expr.args) expr.args = expr.args.map(a => this._optimizeExpr(a));

        // Get operator - AST may use 'op' or 'operator'
        const op = expr.op || expr.operator;

        // Constant folding for binary operations
        if (expr.type === 'binary' &&
            this._isConstantExpr(expr.left) &&
            this._isConstantExpr(expr.right)) {
            return this._foldBinary(expr);
        }

        // Boolean simplification
        if (expr.type === 'binary' && op === 'AND') {
            if (this._isTrueExpr(expr.right)) return expr.left;
            if (this._isTrueExpr(expr.left)) return expr.right;
            if (this._isFalseExpr(expr.left) || this._isFalseExpr(expr.right)) {
                return { type: 'literal', value: false };
            }
        }
        if (expr.type === 'binary' && op === 'OR') {
            if (this._isFalseExpr(expr.right)) return expr.left;
            if (this._isFalseExpr(expr.left)) return expr.right;
            if (this._isTrueExpr(expr.left) || this._isTrueExpr(expr.right)) {
                return { type: 'literal', value: true };
            }
        }

        return expr;
    }

    /**
     * Check if expression is a constant
     */
    _isConstantExpr(expr) {
        return expr && ['literal', 'number', 'string'].includes(expr.type);
    }

    /**
     * Check if expression is TRUE
     */
    _isTrueExpr(expr) {
        return expr?.type === 'literal' && expr.value === true;
    }

    /**
     * Check if expression is FALSE
     */
    _isFalseExpr(expr) {
        return expr?.type === 'literal' && expr.value === false;
    }

    /**
     * Fold binary constant expression
     * @param {Object} expr - Binary expression with constant operands
     * @returns {Object} Literal result
     */
    _foldBinary(expr) {
        const left = this._getConstantValueExpr(expr.left);
        const right = this._getConstantValueExpr(expr.right);

        // Get operator - AST may use 'op' or 'operator'
        const op = expr.op || expr.operator;

        let result;
        switch (op) {
            case '+': result = left + right; break;
            case '-': result = left - right; break;
            case '*': result = left * right; break;
            case '/': result = right !== 0 ? left / right : null; break;
            case '%': result = left % right; break;
            case '=': case '==': result = left === right; break;
            case '!=': case '<>': result = left !== right; break;
            case '<': result = left < right; break;
            case '>': result = left > right; break;
            case '<=': result = left <= right; break;
            case '>=': result = left >= right; break;
            default: return expr;  // Can't fold
        }

        return { type: 'literal', value: result };
    }

    /**
     * Get constant value from expression
     */
    _getConstantValueExpr(expr) {
        if (expr.type === 'number') return expr.value;
        if (expr.type === 'string') return expr.value;
        if (expr.type === 'literal') return expr.value;
        return null;
    }

    /**
     * Extract range predicates from WHERE clause for statistics-based pruning
     * @param {Object} where - WHERE clause AST
     * @returns {Array} Array of predicate objects
     */
    _extractRangePredicates(where) {
        const predicates = [];
        this._collectRangePredicates(where, predicates);
        return predicates;
    }

    /**
     * Recursively collect range predicates
     */
    _collectRangePredicates(expr, predicates) {
        if (!expr) return;

        // Get operator - AST uses 'op' or 'operator'
        const op = expr.op || expr.operator;

        // Handle AND - recurse both sides
        if (expr.type === 'binary' && op === 'AND') {
            this._collectRangePredicates(expr.left, predicates);
            this._collectRangePredicates(expr.right, predicates);
            return;
        }

        // Range operators (normalize '==' to '=')
        const normalizedOp = op === '==' ? '=' : op;
        if (['>', '<', '>=', '<=', '=', '!=', '<>'].includes(normalizedOp)) {
            // Column on left, constant on right
            if (this._isColumnRefExpr(expr.left) && this._isConstantExpr(expr.right)) {
                predicates.push({
                    column: this._getColumnNameExpr(expr.left),
                    operator: normalizedOp,
                    value: this._getConstantValueExpr(expr.right)
                });
            }
            // Constant on left, column on right - flip operator
            else if (this._isConstantExpr(expr.left) && this._isColumnRefExpr(expr.right)) {
                predicates.push({
                    column: this._getColumnNameExpr(expr.right),
                    operator: this._flipOperatorExpr(normalizedOp),
                    value: this._getConstantValueExpr(expr.left)
                });
            }
        }

        // BETWEEN clause
        if (expr.type === 'between' && expr.expr) {
            const col = this._getColumnNameExpr(expr.expr);
            if (col && expr.low && expr.high) {
                predicates.push({
                    column: col,
                    operator: '>=',
                    value: this._getConstantValueExpr(expr.low)
                });
                predicates.push({
                    column: col,
                    operator: '<=',
                    value: this._getConstantValueExpr(expr.high)
                });
            }
        }
    }

    /**
     * Flip comparison operator (for constant on left side)
     */
    _flipOperatorExpr(op) {
        const flips = { '>': '<', '<': '>', '>=': '<=', '<=': '>=' };
        return flips[op] || op;
    }

    /**
     * Check if expression is a column reference
     */
    _isColumnRefExpr(expr) {
        return expr && (expr.type === 'column' || expr.type === 'identifier');
    }

    /**
     * Get column name from expression
     */
    _getColumnNameExpr(expr) {
        if (expr.type === 'column') return expr.name || expr.column;
        if (expr.type === 'identifier') return expr.name || expr.value;
        return null;
    }

    /**
     * Check if a fragment can be pruned based on statistics and predicates
     * @param {Object} fragmentStats - Column statistics for the fragment
     * @param {Array} predicates - Extracted predicates from WHERE clause
     * @returns {boolean} True if fragment can be safely skipped
     */
    _canPruneFragment(fragmentStats, predicates) {
        for (const pred of predicates) {
            const stats = fragmentStats[pred.column];
            if (!stats) continue;  // No stats for this column

            const { min, max, nullCount, rowCount } = stats;

            // All nulls - can't satisfy any comparison
            if (nullCount === rowCount) return true;

            switch (pred.operator) {
                case '>':
                    // If max <= value, no rows can satisfy > value
                    if (max <= pred.value) return true;
                    break;
                case '>=':
                    if (max < pred.value) return true;
                    break;
                case '<':
                    if (min >= pred.value) return true;
                    break;
                case '<=':
                    if (min > pred.value) return true;
                    break;
                case '=':
                    // If value outside [min, max], no match possible
                    if (pred.value < min || pred.value > max) return true;
                    break;
                case '!=':
                case '<>':
                    // Can only prune if all values are the same and equal to pred.value
                    if (min === max && min === pred.value) return true;
                    break;
            }
        }

        return false;  // Cannot prune
    }

    /**
     * Execute EXPLAIN query - return query plan without executing
     * @param {Object} ast - Parsed AST of the inner query
     * @returns {Object} Plan information
     */
    _explainQuery(ast) {
        const plan = {
            type: ast.type,
            tables: [],
            predicates: [],
            optimizations: []
        };

        // Collect table info
        if (ast.from) {
            plan.tables.push({
                name: ast.from.name || ast.from.table,
                alias: ast.from.alias
            });
        }

        // Collect joined tables
        if (ast.joins) {
            for (const join of ast.joins) {
                plan.tables.push({
                    name: join.table?.name || join.table?.table,
                    alias: join.table?.alias,
                    joinType: join.type
                });
            }
        }

        // Extract predicates from WHERE
        if (ast.where) {
            plan.predicates = this._extractRangePredicates(ast.where);
        }

        // Identify optimizations
        if (ast.where) {
            plan.optimizations.push('PREDICATE_PUSHDOWN');
        }
        if (ast.groupBy) {
            plan.optimizations.push('AGGREGATE');
        }
        if (ast.orderBy) {
            plan.optimizations.push('SORT');
        }
        if (ast.limit) {
            plan.optimizations.push('LIMIT_PUSHDOWN');
        }

        return {
            columns: ['Plan'],
            rows: [[JSON.stringify(plan, null, 2)]],
            total: 1
        };
    }

    // ========================================================================
    // Phase 10: Memory Table CRUD Operations
    // ========================================================================

    /**
     * Execute CREATE TABLE - creates an in-memory table
     * @param {Object} ast - Parsed CREATE TABLE AST
     * @returns {Object} Result with success flag
     */
    _executeCreateTable(ast) {
        const tableName = (ast.table || ast.name || '').toLowerCase();

        if (!tableName) {
            throw new Error('CREATE TABLE requires a table name');
        }

        // Check if table already exists (memory or remote)
        if (this.memoryTables.has(tableName) || this.tables.has(tableName)) {
            if (ast.ifNotExists) {
                return { success: true, existed: true, table: tableName };
            }
            throw new Error(`Table '${tableName}' already exists`);
        }

        // Build schema from AST columns
        const schema = (ast.columns || []).map(col => ({
            name: col.name,
            dataType: col.dataType || col.type || 'TEXT',
            primaryKey: col.primaryKey || false
        }));

        if (schema.length === 0) {
            throw new Error('CREATE TABLE requires at least one column');
        }

        // Create and store the memory table
        const table = new MemoryTable(tableName, schema);
        this.memoryTables.set(tableName, table);

        return {
            success: true,
            table: tableName,
            columns: schema.map(c => c.name)
        };
    }

    /**
     * Execute DROP TABLE - removes an in-memory table
     * @param {Object} ast - Parsed DROP TABLE AST
     * @returns {Object} Result with success flag
     */
    _executeDropTable(ast) {
        const tableName = (ast.table || ast.name || '').toLowerCase();

        if (!this.memoryTables.has(tableName)) {
            if (ast.ifExists) {
                return { success: true, existed: false, table: tableName };
            }
            throw new Error(`Memory table '${tableName}' not found`);
        }

        this.memoryTables.delete(tableName);
        return { success: true, table: tableName };
    }

    /**
     * Execute INSERT - adds rows to a memory table
     * @param {Object} ast - Parsed INSERT AST
     * @returns {Object} Result with inserted count
     */
    _executeInsert(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found. Use CREATE TABLE first.`);
        }

        // Get column names to insert into (use table columns if not specified)
        const insertCols = ast.columns || table.columns;
        let inserted = 0;

        // Process each row from VALUES clause
        for (const astRow of (ast.rows || ast.values || [])) {
            const row = new Array(table.columns.length).fill(null);

            insertCols.forEach((colName, i) => {
                const colIdx = table._columnIndex.get(
                    (typeof colName === 'string' ? colName : colName.name || colName).toLowerCase()
                );
                if (colIdx !== undefined && i < astRow.length) {
                    // Handle AST value nodes or raw values
                    const val = astRow[i];
                    row[colIdx] = val?.value !== undefined ? val.value : val;
                }
            });

            table.rows.push(row);
            inserted++;
        }

        return {
            success: true,
            inserted,
            total: table.rows.length
        };
    }

    /**
     * Execute UPDATE - modifies rows in a memory table
     * @param {Object} ast - Parsed UPDATE AST
     * @returns {Object} Result with updated count
     */
    _executeUpdate(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found`);
        }

        // Build column data for WHERE expression evaluation
        const columnData = {};
        table.columns.forEach((col, idx) => {
            columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
        });

        // Create executor for expression evaluation
        const executor = new SQLExecutor({ columnNames: table.columns });
        let updated = 0;

        // Process each row
        for (let i = 0; i < table.rows.length; i++) {
            // Check WHERE condition (if present)
            const matches = !ast.where || executor._evaluateInMemoryExpr(ast.where, columnData, i);

            if (matches) {
                // Apply SET assignments
                for (const assignment of (ast.assignments || ast.set || [])) {
                    const colName = (assignment.column || assignment.name || '').toLowerCase();
                    const colIdx = table._columnIndex.get(colName);

                    if (colIdx !== undefined) {
                        const val = assignment.value;
                        table.rows[i][colIdx] = val?.value !== undefined ? val.value : val;
                    }
                }
                updated++;
            }
        }

        return { success: true, updated };
    }

    /**
     * Execute DELETE - removes rows from a memory table
     * @param {Object} ast - Parsed DELETE AST
     * @returns {Object} Result with deleted count
     */
    _executeDelete(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found`);
        }

        const originalCount = table.rows.length;

        if (ast.where) {
            // Build column data for WHERE expression evaluation
            const columnData = {};
            table.columns.forEach((col, idx) => {
                columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
            });

            // Create executor for expression evaluation
            const executor = new SQLExecutor({ columnNames: table.columns });

            // Keep rows that DON'T match the WHERE condition
            table.rows = table.rows.filter((_, i) =>
                !executor._evaluateInMemoryExpr(ast.where, columnData, i)
            );
        } else {
            // DELETE without WHERE = truncate
            table.rows = [];
        }

        return {
            success: true,
            deleted: originalCount - table.rows.length,
            remaining: table.rows.length
        };
    }
}


export { LanceDatabase };
