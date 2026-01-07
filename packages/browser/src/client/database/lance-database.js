/**
 * LanceDatabase - Multi-table SQL database
 */

import { SQLLexer } from '../sql/lexer.js';
import { SQLParser } from '../sql/parser.js';
import { SQLExecutor } from '../sql/executor.js';

// Extracted modules
import * as JoinsModule from './lance-db-joins.js';
import * as MemoryModule from './lance-db-memory.js';
import * as OptimizerModule from './lance-db-optimizer.js';

// Re-export MemoryTable for external use
export { MemoryTable } from './lance-db-memory.js';

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
     */
    register(name, dataset) {
        this.tables.set(name, dataset);
    }

    /**
     * Register a remote dataset by URL
     */
    async registerRemote(name, url, options = {}) {
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
        const actualName = this.aliases.get(name) || name;
        const table = this.tables.get(actualName);
        if (!table) {
            throw new Error(`Table '${name}' not found. Did you forget to register it?`);
        }
        return table;
    }

    /**
     * Execute SQL query
     */
    async executeSQL(sql) {
        // Check plan cache first
        const cachedPlan = OptimizerModule.getCachedPlan(this, sql);
        let ast;

        if (cachedPlan) {
            ast = cachedPlan;
        } else {
            const lexer = new SQLLexer(sql);
            const tokens = lexer.tokenize();
            const parser = new SQLParser(tokens);
            ast = parser.parse();

            if (ast.type !== 'EXPLAIN') {
                OptimizerModule.setCachedPlan(this, sql, ast);
            }
        }

        // Handle EXPLAIN
        if (ast.type === 'EXPLAIN') {
            return OptimizerModule.explainQuery(this, ast.statement);
        }

        // Handle memory table operations
        if (ast.type === 'CREATE_TABLE') {
            return MemoryModule.executeCreateTable(this, ast);
        }
        if (ast.type === 'DROP_TABLE') {
            return MemoryModule.executeDropTable(this, ast);
        }
        if (ast.type === 'INSERT') {
            return MemoryModule.executeInsert(this, ast);
        }
        if (ast.type === 'UPDATE') {
            return MemoryModule.executeUpdate(this, ast);
        }
        if (ast.type === 'DELETE') {
            return MemoryModule.executeDelete(this, ast);
        }

        if (ast.type === 'SET_OPERATION') {
            return this._executeSetOperation(ast);
        }

        if (ast.type !== 'SELECT') {
            throw new Error('Only SELECT queries are supported in LanceDatabase');
        }

        // Handle CTEs
        if (ast.ctes && ast.ctes.length > 0) {
            return this._executeWithCTEs(ast);
        }

        // No joins - simple single-table query
        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }

        // Multi-table query with JOINs
        return JoinsModule.executeJoin(this, ast);
    }

    /**
     * Execute query with CTEs
     */
    async _executeWithCTEs(ast) {
        const cteExecutor = new SQLExecutor({ columnNames: [] });
        cteExecutor.setDatabase(this);
        await cteExecutor.materializeCTEs(ast.ctes, this);

        const mainTableName = ast.from?.name?.toLowerCase() || ast.from?.table?.toLowerCase();
        if (mainTableName && cteExecutor._cteResults.has(mainTableName)) {
            return cteExecutor._executeOnInMemoryData(ast, cteExecutor._cteResults.get(mainTableName));
        }

        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }
        return JoinsModule.executeJoin(this, ast);
    }

    /**
     * Execute SET operation (UNION, INTERSECT, EXCEPT)
     */
    async _executeSetOperation(ast) {
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

        // Apply ORDER BY
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

        // Apply LIMIT/OFFSET
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
        const executor = new SQLExecutor(dataset);
        return executor.execute(ast);
    }

    /**
     * Extract column name from expression
     */
    _extractColumnFromExpr(expr, expectedTable) {
        if (expr.type === 'column') {
            if (expr.table && expr.table !== expectedTable) {
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

    // Optimizer delegations
    clearPlanCache() {
        this._planCache.clear();
    }

    getPlanCacheStats() {
        return OptimizerModule.getPlanCacheStats(this);
    }

    _optimizeExpr(expr) {
        return OptimizerModule.optimizeExpr(expr);
    }

    _extractRangePredicates(where) {
        return OptimizerModule.extractRangePredicates(where);
    }

    _canPruneFragment(fragmentStats, predicates) {
        return OptimizerModule.canPruneFragment(fragmentStats, predicates);
    }

    // Join helper delegations (for backward compatibility)
    _findColumnIndex(columns, columnName) {
        return JoinsModule.findColumnIndex(columns, columnName);
    }

    _filterToSQL(expr) {
        return JoinsModule.filterToSQL(expr);
    }

    _applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
        return JoinsModule.applyProjection(rows, allColumns, projection, leftAlias, rightAlias);
    }

    _buildInClause(column, keys) {
        return JoinsModule.buildInClause(column, keys);
    }

    _hashJoinWithInMemoryLeft(leftResult, rightDataset, ast, context) {
        return JoinsModule.hashJoinWithInMemoryLeft(this, leftResult, rightDataset, ast, context);
    }

    _appendWhereClause(sql, clause) {
        return JoinsModule.appendWhereClause(sql, clause);
    }
}


export { LanceDatabase };
