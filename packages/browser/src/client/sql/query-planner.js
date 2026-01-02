/**
 * QueryPlanner - Cost-based query plan generation
 */

import { StatisticsManager, statisticsManager } from './statistics-manager.js';
import { CostModel } from './cost-model.js';
import * as SingleTable from './planner-single.js';

class QueryPlanner {
    constructor() {
        this.debug = true;
    }

    /**
     * Generate physical execution plan from logical AST
     * @param {Object} ast - Parsed SQL AST
     * @param {Object} context - Table names and aliases
     * @returns {Object} Physical execution plan
     */
    plan(ast, context) {
        const { leftTableName, leftAlias, rightTableName, rightAlias } = context;

        const columnAnalysis = this._analyzeColumns(ast, context);
        const filterAnalysis = this._analyzeFilters(ast, context);
        const fetchEstimate = this._estimateFetchSize(ast, filterAnalysis);

        const plan = {
            leftScan: {
                table: leftTableName,
                alias: leftAlias,
                columns: columnAnalysis.left.all,
                filters: filterAnalysis.left,
                limit: fetchEstimate.left,
                purpose: {
                    join: columnAnalysis.left.join,
                    where: columnAnalysis.left.where,
                    result: columnAnalysis.left.result
                }
            },
            rightScan: {
                table: rightTableName,
                alias: rightAlias,
                columns: columnAnalysis.right.all,
                filters: filterAnalysis.right,
                filterByJoinKeys: true,
                purpose: {
                    join: columnAnalysis.right.join,
                    where: columnAnalysis.right.where,
                    result: columnAnalysis.right.result
                }
            },
            join: {
                type: ast.joins[0].type,
                leftKey: columnAnalysis.joinKeys.left,
                rightKey: columnAnalysis.joinKeys.right,
                algorithm: 'HASH_JOIN'
            },
            projection: columnAnalysis.resultColumns,
            limit: ast.limit || null,
            offset: ast.offset || 0
        };

        if (this.debug) {
            this._logPlan(plan, ast);
        }

        return plan;
    }

    /**
     * Generate optimized plan for single-table queries
     */
    planSingleTable(ast) {
        return SingleTable.planSingleTable(this, ast);
    }

    _analyzeColumns(ast, context) {
        const { leftAlias, rightAlias } = context;

        const left = { join: new Set(), where: new Set(), result: new Set(), all: [] };
        const right = { join: new Set(), where: new Set(), result: new Set(), all: [] };

        for (const item of ast.columns) {
            if (item.type === 'star') {
                left.result.add('*');
                right.result.add('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const table = col.table || null;
                const column = col.column;

                if (!table || table === leftAlias) {
                    left.result.add(column);
                }
                if (!table || table === rightAlias) {
                    right.result.add(column);
                }
            }
        }

        const join = ast.joins[0];
        const joinKeys = this._extractJoinKeys(join.on, leftAlias, rightAlias);

        if (joinKeys.left) left.join.add(joinKeys.left);
        if (joinKeys.right) right.join.add(joinKeys.right);

        if (ast.where) {
            this._extractWhereColumns(ast.where, leftAlias, rightAlias, left.where, right.where);
        }

        left.all = [...new Set([...left.join, ...left.where, ...left.result])];
        right.all = [...new Set([...right.join, ...right.where, ...right.result])];

        if (left.result.has('*')) left.all = ['*'];
        if (right.result.has('*')) right.all = ['*'];

        const resultColumns = [];
        for (const item of ast.columns) {
            if (item.type === 'star') {
                resultColumns.push('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const alias = item.alias || `${col.table || ''}.${col.column}`.replace(/^\./, '');
                resultColumns.push({ table: col.table, column: col.column, alias });
            }
        }

        return { left, right, joinKeys, resultColumns };
    }

    _extractJoinKeys(onExpr, leftAlias, rightAlias) {
        if (!onExpr || onExpr.type !== 'binary') {
            return { left: null, right: null };
        }

        const leftCol = onExpr.left;
        const rightCol = onExpr.right;
        let leftKey = null, rightKey = null;

        if (leftCol.type === 'column') {
            if (!leftCol.table || leftCol.table === leftAlias) {
                leftKey = leftCol.column;
            } else if (leftCol.table === rightAlias) {
                rightKey = leftCol.column;
            }
        }

        if (rightCol.type === 'column') {
            if (!rightCol.table || rightCol.table === leftAlias) {
                leftKey = rightCol.column;
            } else if (rightCol.table === rightAlias) {
                rightKey = rightCol.column;
            }
        }

        return { left: leftKey, right: rightKey };
    }

    _extractWhereColumns(expr, leftAlias, rightAlias, leftCols, rightCols) {
        if (!expr) return;

        if (expr.type === 'column') {
            const table = expr.table;
            const column = expr.column;

            if (!table || table === leftAlias) leftCols.add(column);
            else if (table === rightAlias) rightCols.add(column);
        } else if (expr.type === 'binary') {
            this._extractWhereColumns(expr.left, leftAlias, rightAlias, leftCols, rightCols);
            this._extractWhereColumns(expr.right, leftAlias, rightAlias, leftCols, rightCols);
        } else if (expr.type === 'unary') {
            this._extractWhereColumns(expr.expr, leftAlias, rightAlias, leftCols, rightCols);
        }
    }

    _analyzeFilters(ast, context) {
        const { leftAlias, rightAlias } = context;
        const left = [], right = [], join = [];

        if (ast.where) {
            this._separateFilters(ast.where, leftAlias, rightAlias, left, right, join);
        }

        return { left, right, join };
    }

    _separateFilters(expr, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters) {
        if (!expr) return;

        if (expr.type === 'binary' && expr.op === 'AND') {
            this._separateFilters(expr.left, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            this._separateFilters(expr.right, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            return;
        }

        const tables = this._getReferencedTables(expr, leftAlias, rightAlias);

        if (tables.size === 1) {
            if (tables.has(leftAlias)) leftFilters.push(expr);
            else if (tables.has(rightAlias)) rightFilters.push(expr);
        } else if (tables.size > 1) {
            joinFilters.push(expr);
        }
    }

    _getReferencedTables(expr, leftAlias, rightAlias) {
        const tables = new Set();

        const walk = (e) => {
            if (!e) return;

            if (e.type === 'column') {
                const table = e.table;
                if (!table) {
                    tables.add(leftAlias);
                    tables.add(rightAlias);
                } else if (table === leftAlias) {
                    tables.add(leftAlias);
                } else if (table === rightAlias) {
                    tables.add(rightAlias);
                }
            } else if (e.type === 'binary') {
                walk(e.left);
                walk(e.right);
            } else if (e.type === 'unary') {
                walk(e.operand);
            } else if (e.type === 'call') {
                for (const arg of e.args || []) walk(arg);
            } else if (e.type === 'in') {
                walk(e.expr);
                for (const v of e.values || []) walk(v);
            } else if (e.type === 'between') {
                walk(e.expr);
                walk(e.low);
                walk(e.high);
            } else if (e.type === 'like') {
                walk(e.expr);
                walk(e.pattern);
            }
        };

        walk(expr);
        return tables;
    }

    _estimateFetchSize(ast, filterAnalysis) {
        const requestedLimit = ast.limit || 1000;
        const leftSelectivity = filterAnalysis.left.length > 0 ? 0.5 : 1.0;
        const joinSelectivity = 0.7;
        const safetyFactor = 2.5;

        const leftFetch = Math.ceil(
            requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
        );

        return {
            left: Math.min(leftFetch, 10000),
            right: null
        };
    }

    _logPlan(plan, ast) {
        console.log('\n' + '='.repeat(60));
        console.log('📋 QUERY EXECUTION PLAN');
        console.log('='.repeat(60));

        console.log('\n🔍 Original Query:');
        console.log(`  SELECT: ${ast.columns.length} columns`);
        console.log(`  FROM: ${plan.leftScan.table} AS ${plan.leftScan.alias}`);
        console.log(`  JOIN: ${plan.rightScan.table} AS ${plan.rightScan.alias}`);
        console.log(`  WHERE: ${ast.where ? 'yes' : 'no'}`);
        console.log(`  LIMIT: ${ast.limit || 'none'}`);

        console.log('\n📊 Physical Plan:');

        console.log('\n  Step 1: SCAN LEFT TABLE');
        console.log(`    Table: ${plan.leftScan.table}`);
        console.log(`    Columns: [${plan.leftScan.columns.join(', ')}]`);
        console.log(`    Filters: ${plan.leftScan.filters.length} pushed down`);
        plan.leftScan.filters.forEach((f, i) => {
            console.log(`      ${i + 1}. ${this._formatFilter(f)}`);
        });
        console.log(`    Limit: ${plan.leftScan.limit} rows`);

        console.log('\n  Step 2: BUILD HASH TABLE');
        console.log(`    Index by: ${plan.join.leftKey}`);

        console.log('\n  Step 3: SCAN RIGHT TABLE');
        console.log(`    Table: ${plan.rightScan.table}`);
        console.log(`    Columns: [${plan.rightScan.columns.join(', ')}]`);
        console.log(`    Filters: ${plan.rightScan.filters.length} pushed down`);
        console.log(`    Dynamic filter: ${plan.join.rightKey} IN (keys from left)`);

        console.log('\n  Step 4: HASH JOIN');
        console.log(`    Algorithm: ${plan.join.algorithm}`);
        console.log(`    Condition: ${plan.join.leftKey} = ${plan.join.rightKey}`);

        console.log('\n  Step 5: PROJECT');
        console.log(`    Result columns: ${plan.projection.length}`);

        console.log('\n  Step 6: LIMIT');
        console.log(`    Rows: ${plan.limit || 'none'}`);

        console.log('\n' + '='.repeat(60) + '\n');
    }

    _formatFilter(expr) {
        if (!expr) return 'null';

        if (expr.type === 'binary') {
            const left = this._formatFilter(expr.left);
            const right = this._formatFilter(expr.right);
            return `${left} ${expr.op} ${right}`;
        } else if (expr.type === 'column') {
            return expr.table ? `${expr.table}.${expr.column}` : expr.column;
        } else if (expr.type === 'literal') {
            return JSON.stringify(expr.value);
        } else if (expr.type === 'call') {
            const args = (expr.args || []).map(a => this._formatFilter(a)).join(', ');
            return `${expr.name}(${args})`;
        }

        return JSON.stringify(expr);
    }
}

export { StatisticsManager, CostModel, QueryPlanner, statisticsManager };
