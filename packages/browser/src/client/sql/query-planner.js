import { StatisticsManager, statisticsManager } from './statistics-manager.js';
import { CostModel } from './cost-model.js';
import * as SingleTable from './planner-single.js';

class QueryPlanner {
    plan(ast, context) {
        const { leftTableName, leftAlias, rightTableName, rightAlias } = context;

        const columnAnalysis = this._analyzeColumns(ast, context);
        const filterAnalysis = this._analyzeFilters(ast, context);
        const fetchEstimate = this._estimateFetchSize(ast, filterAnalysis);

        return {
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
    }

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

                if (!table || table === leftAlias) left.result.add(column);
                if (!table || table === rightAlias) right.result.add(column);
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
        if (!onExpr || onExpr.type !== 'binary') return { left: null, right: null };

        const leftCol = onExpr.left;
        const rightCol = onExpr.right;
        let leftKey = null, rightKey = null;

        if (leftCol.type === 'column') {
            if (!leftCol.table || leftCol.table === leftAlias) leftKey = leftCol.column;
            else if (leftCol.table === rightAlias) rightKey = leftCol.column;
        }

        if (rightCol.type === 'column') {
            if (!rightCol.table || rightCol.table === leftAlias) leftKey = rightCol.column;
            else if (rightCol.table === rightAlias) rightKey = rightCol.column;
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

        return { left: Math.min(leftFetch, 10000), right: null };
    }
}

export { StatisticsManager, CostModel, QueryPlanner, statisticsManager };
