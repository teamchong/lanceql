/**
 * SQL Executor - Execute parsed SQL AST
 */

import { TokenType, SQLLexer } from './tokenizer.js';
import { SQLParser } from './parser.js';
import { getGPUTransformerState, getEmbeddingCache } from '../worker-store.js';

function getColumnValue(row, column, tableAliases = {}) {
    if (typeof column === 'string') {
        return row[column];
    }
    // Handle literal values (e.g., WHERE 1 = 0)
    if (column && column.type === 'literal') {
        return column.value;
    }
    if (column && column.table && column.column) {
        // Try table.column directly first (e.g., u.id)
        const fullKey = `${column.table}.${column.column}`;
        if (fullKey in row) return row[fullKey];
        // Try using alias mapping to get actual table name
        const tableName = tableAliases[column.table] || column.table;
        const aliasKey = `${tableName}.${column.column}`;
        if (aliasKey in row) return row[aliasKey];
        // Try just the column name (fallback for non-JOIN queries)
        if (column.column in row) return row[column.column];
    }
    // Handle column without table prefix (e.g., {column: 'id'})
    if (column && column.column && !column.table) {
        if (column.column in row) return row[column.column];
    }
    return undefined;
}

// Flatten joined row for DML with JOIN support (UPDATE...FROM, DELETE...USING)
function flattenJoinedRow(jr) {
    const flat = {};
    for (const [alias, row] of Object.entries(jr)) {
        if (alias === '__idx') continue;
        if (typeof row === 'object' && row !== null) {
            for (const [col, val] of Object.entries(row)) {
                flat[`${alias}.${col}`] = val;
                if (!(col in flat)) flat[col] = val;
            }
        }
    }
    return flat;
}

// Evaluate compound JOIN condition (supports AND/OR and multiple comparison operators)
function evalJoinCondition(condition, leftRow, rightRow, tableAliases) {
    if (!condition) return true;

    // Handle AND/OR compound conditions
    if (condition.op === 'AND') {
        return evalJoinCondition(condition.left, leftRow, rightRow, tableAliases) &&
               evalJoinCondition(condition.right, leftRow, rightRow, tableAliases);
    }
    if (condition.op === 'OR') {
        return evalJoinCondition(condition.left, leftRow, rightRow, tableAliases) ||
               evalJoinCondition(condition.right, leftRow, rightRow, tableAliases);
    }

    // Handle comparison - try both orderings since we don't know which row has which column
    const leftVal = getColumnValue(leftRow, condition.left, tableAliases) ??
                    getColumnValue(rightRow, condition.left, tableAliases);
    const rightVal = getColumnValue(rightRow, condition.right, tableAliases) ??
                     getColumnValue(leftRow, condition.right, tableAliases);

    switch (condition.op) {
        case '=': return leftVal === rightVal;
        case '!=': return leftVal !== rightVal;
        case '<': return leftVal < rightVal;
        case '<=': return leftVal <= rightVal;
        case '>': return leftVal > rightVal;
        case '>=': return leftVal >= rightVal;
        default: return false;
    }
}

function evalWhere(where, row, tableAliases = {}) {
    if (!where) return true;

    // Helper to resolve value - could be literal or column reference
    const resolveValue = (val) => {
        if (val && typeof val === 'object' && (val.table || val.column)) {
            // It's a column reference
            return getColumnValue(row, val, tableAliases);
        }
        return val;
    };

    switch (where.op) {
        case 'AND':
            return evalWhere(where.left, row, tableAliases) && evalWhere(where.right, row, tableAliases);
        case 'OR':
            return evalWhere(where.left, row, tableAliases) || evalWhere(where.right, row, tableAliases);
        case '=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val === compareVal;
        }
        case '!=':
        case '<>': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val !== compareVal;
        }
        case '<': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val < compareVal;
        }
        case '<=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val <= compareVal;
        }
        case '>': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val > compareVal;
        }
        case '>=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val >= compareVal;
        }
        case 'LIKE': {
            const val = getColumnValue(row, where.column, tableAliases);
            const pattern = where.value.replace(/%/g, '.*').replace(/_/g, '.');
            return new RegExp(`^${pattern}$`, 'i').test(val);
        }
        case 'BETWEEN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val >= where.low && val <= where.high;
        }
        case 'IN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return where.values.includes(val);
        }
        case 'NOT IN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return !where.values.includes(val);
        }
        case 'NOT BETWEEN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val < where.low || val > where.high;
        }
        case 'NOT LIKE': {
            const val = getColumnValue(row, where.column, tableAliases);
            const pattern = where.value.replace(/%/g, '.*').replace(/_/g, '.');
            return !new RegExp(`^${pattern}$`, 'i').test(val);
        }
        case 'IS NULL': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val === null || val === undefined;
        }
        case 'IS NOT NULL': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val !== null && val !== undefined;
        }
        case 'IN SUBQUERY': {
            const val = getColumnValue(row, where.column, tableAliases);
            // subqueryValues should be pre-populated before eval
            return where.subqueryValues ? where.subqueryValues.includes(val) : false;
        }
        case 'NOT IN SUBQUERY': {
            const val = getColumnValue(row, where.column, tableAliases);
            return where.subqueryValues ? !where.subqueryValues.includes(val) : true;
        }
        case 'EXISTS': {
            // existsResult should be pre-populated before eval
            return where.existsResult === true;
        }
        case 'NOT EXISTS': {
            return where.existsResult === false;
        }
        default:
            return true;
    }
}

// Pre-execute subqueries in WHERE clause and populate subqueryValues
async function preExecuteSubqueries(where, db) {
    if (!where) return;

    if (where.op === 'AND' || where.op === 'OR') {
        await preExecuteSubqueries(where.left, db);
        await preExecuteSubqueries(where.right, db);
        return;
    }

    if (where.op === 'IN SUBQUERY' || where.op === 'NOT IN SUBQUERY') {
        // Execute the subquery and extract values
        const subAst = where.subquery;
        const subRows = await db.select(subAst.table);

        // Get the first column from each row
        const firstCol = subAst.columns[0];
        const colName = firstCol.type === 'column'
            ? (firstCol.value.column || firstCol.value)
            : (typeof firstCol === 'string' ? firstCol : firstCol.value);

        where.subqueryValues = subRows.map(row => row[colName]);
    }

    if (where.op === 'EXISTS' || where.op === 'NOT EXISTS') {
        // Execute the subquery and check if any rows exist
        const result = await executeAST(db, where.subquery);
        where.existsResult = result.rows.length > 0;
    }
}

// Calculate aggregate function value
function calculateAggregate(func, arg, rows) {
    if (rows.length === 0) return func === 'count' ? 0 : null;

    const colName = arg === '*' ? null : (typeof arg === 'string' ? arg : (arg.column || arg));

    switch (func) {
        case 'count':
            if (arg === '*') return rows.length;
            return rows.filter(r => r[colName] != null).length;
        case 'sum': {
            let sum = 0;
            for (const row of rows) {
                const val = row[colName];
                if (typeof val === 'number') sum += val;
            }
            return sum;
        }
        case 'avg': {
            let sum = 0, count = 0;
            for (const row of rows) {
                const val = row[colName];
                if (typeof val === 'number') {
                    sum += val;
                    count++;
                }
            }
            return count > 0 ? sum / count : null;
        }
        case 'min': {
            let min = Infinity;
            for (const row of rows) {
                const val = row[colName];
                if (val != null && val < min) min = val;
            }
            return min === Infinity ? null : min;
        }
        case 'max': {
            let max = -Infinity;
            for (const row of rows) {
                const val = row[colName];
                if (val != null && val > max) max = val;
            }
            return max === -Infinity ? null : max;
        }
        case 'stddev': case 'stddev_samp': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length < 2) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
            return Math.sqrt(variance);
        }
        case 'stddev_pop': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length === 0) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
            return Math.sqrt(variance);
        }
        case 'variance': case 'var_samp': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length < 2) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            return vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
        }
        case 'var_pop': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length === 0) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            return vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
        }
        case 'median': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number').sort((a, b) => a - b);
            if (vals.length === 0) return null;
            const mid = Math.floor(vals.length / 2);
            return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
        }
        case 'string_agg': case 'group_concat': {
            const actualCol = typeof arg === 'object' && arg.column ? arg.column : colName;
            const separator = typeof arg === 'object' && arg.separator != null ? arg.separator : ',';
            const vals = rows.map(r => r[actualCol]).filter(v => v != null).map(String);
            return vals.join(separator);
        }
        default:
            return null;
    }
}

// Evaluate arithmetic expression for a single row
function evaluateArithmeticExpr(expr, row, tableAliases = {}, excluded = null) {
    if (!expr) return null;

    switch (expr.type) {
        case 'literal':
            return expr.value;
        case 'column': {
            // Handle EXCLUDED.column for UPSERT
            const colName = typeof expr.value === 'string' ? expr.value : expr.value?.column;
            if (colName && colName.toUpperCase().startsWith('EXCLUDED.')) {
                const field = colName.substring(9);
                return excluded?.[field] ?? null;
            }
            return getColumnValue(row, expr.value, tableAliases);
        }
        case 'function':
            return evaluateScalarFunction(expr.func, expr.args, row, tableAliases);
        case 'array_literal':
            return expr.elements.map(el => evaluateArithmeticExpr(el, row, tableAliases));
        case 'subscript': {
            const arr = evaluateArithmeticExpr(expr.array, row, tableAliases);
            const idx = evaluateArithmeticExpr(expr.index, row, tableAliases);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null; // SQL uses 1-based indexing
        }
        case 'arithmetic':
            if (expr.op === 'unary-') {
                const operand = evaluateArithmeticExpr(expr.operand, row, tableAliases);
                return operand != null ? -operand : null;
            }
            if (expr.op === 'unary~') {
                const operand = evaluateArithmeticExpr(expr.operand, row, tableAliases);
                return operand != null ? ~(operand | 0) : null;
            }
            const left = evaluateArithmeticExpr(expr.left, row, tableAliases);
            const right = evaluateArithmeticExpr(expr.right, row, tableAliases);
            if (left == null || right == null) return null;
            switch (expr.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                // Bitwise operators
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        default:
            return null;
    }
}

// Evaluate scalar function for a single row
function evaluateScalarFunction(func, args, row, tableAliases = {}) {
    const evalArg = (arg) => {
        if (!arg) return null;
        if (arg.type === 'literal') return arg.value;
        if (arg.type === 'column') return getColumnValue(row, arg.value, tableAliases);
        if (arg.type === 'function') return evaluateScalarFunction(arg.func, arg.args, row, tableAliases);
        if (arg.type === 'array_literal') return arg.elements.map(el => evalArg(el));
        if (arg.type === 'subscript') {
            const arr = evalArg(arg.array);
            const idx = evalArg(arg.index);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null; // SQL 1-indexed
        }
        if (arg.type === 'arithmetic') {
            if (arg.op === 'unary-') return -evalArg(arg.operand);
            if (arg.op === 'unary~') return ~(evalArg(arg.operand) | 0);
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        }
        if (arg.type === 'comparison') {
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '=': return left === right;
                case '!=': case '<>': return left !== right;
                case '<': return left < right;
                case '<=': return left <= right;
                case '>': return left > right;
                case '>=': return left >= right;
                default: return false;
            }
        }
        return null;
    };

    switch (func) {
        // COALESCE - return first non-null
        case 'coalesce': {
            for (const arg of args) {
                const val = evalArg(arg);
                if (val !== null && val !== undefined) return val;
            }
            return null;
        }
        // NULLIF - return null if args are equal
        case 'nullif': {
            const a = evalArg(args[0]);
            const b = evalArg(args[1]);
            return a === b ? null : a;
        }
        // String functions
        case 'upper':
            return String(evalArg(args[0]) ?? '').toUpperCase();
        case 'lower':
            return String(evalArg(args[0]) ?? '').toLowerCase();
        case 'length':
            return String(evalArg(args[0]) ?? '').length;
        case 'substr':
        case 'substring': {
            const str = String(evalArg(args[0]) ?? '');
            const start = (evalArg(args[1]) ?? 1) - 1; // SQL is 1-indexed
            const len = args[2] ? evalArg(args[2]) : undefined;
            return len !== undefined ? str.substr(start, len) : str.substr(start);
        }
        case 'trim':
            return String(evalArg(args[0]) ?? '').trim();
        case 'ltrim':
            return String(evalArg(args[0]) ?? '').trimStart();
        case 'rtrim':
            return String(evalArg(args[0]) ?? '').trimEnd();
        case 'concat':
            return args.map(a => String(evalArg(a) ?? '')).join('');
        case 'replace': {
            const str = String(evalArg(args[0]) ?? '');
            const from = String(evalArg(args[1]) ?? '');
            const to = String(evalArg(args[2]) ?? '');
            return str.split(from).join(to);
        }
        // Math functions
        case 'abs':
            return Math.abs(evalArg(args[0]) ?? 0);
        case 'round': {
            const val = evalArg(args[0]) ?? 0;
            const decimals = args[1] ? evalArg(args[1]) : 0;
            const factor = Math.pow(10, decimals);
            return Math.round(val * factor) / factor;
        }
        case 'ceil':
        case 'ceiling':
            return Math.ceil(evalArg(args[0]) ?? 0);
        case 'floor':
            return Math.floor(evalArg(args[0]) ?? 0);
        case 'mod':
            return (evalArg(args[0]) ?? 0) % (evalArg(args[1]) ?? 1);
        case 'power':
        case 'pow':
            return Math.pow(evalArg(args[0]) ?? 0, evalArg(args[1]) ?? 1);
        case 'sqrt':
            return Math.sqrt(evalArg(args[0]) ?? 0);
        case 'truncate':
        case 'trunc': {
            const val = evalArg(args[0]) ?? 0;
            const scale = args[1] ? evalArg(args[1]) : 0;
            if (scale === 0) return Math.trunc(val);
            const factor = Math.pow(10, scale);
            return Math.trunc(val * factor) / factor;
        }
        case 'sign': {
            const v = evalArg(args[0]);
            if (v == null) return null;
            return v > 0 ? 1 : v < 0 ? -1 : 0;
        }
        case 'log':
        case 'ln':
            return Math.log(evalArg(args[0]) ?? 0);
        case 'log10':
            return Math.log10(evalArg(args[0]) ?? 0);
        case 'exp':
            return Math.exp(evalArg(args[0]) ?? 0);
        case 'sin':
            return Math.sin(evalArg(args[0]) ?? 0);
        case 'cos':
            return Math.cos(evalArg(args[0]) ?? 0);
        case 'tan':
            return Math.tan(evalArg(args[0]) ?? 0);
        case 'asin':
            return Math.asin(evalArg(args[0]) ?? 0);
        case 'acos':
            return Math.acos(evalArg(args[0]) ?? 0);
        case 'atan':
            return Math.atan(evalArg(args[0]) ?? 0);
        case 'atan2':
            return Math.atan2(evalArg(args[0]) ?? 0, evalArg(args[1]) ?? 0);
        case 'pi':
            return Math.PI;
        case 'random':
        case 'rand':
            return Math.random();
        case 'degrees':
            return (evalArg(args[0]) ?? 0) * (180 / Math.PI);
        case 'radians':
            return (evalArg(args[0]) ?? 0) * (Math.PI / 180);

        // ========== Conditional Functions ==========
        case 'greatest': {
            const values = args.map(evalArg).filter(v => v != null);
            return values.length ? Math.max(...values) : null;
        }
        case 'least': {
            const values = args.map(evalArg).filter(v => v != null);
            return values.length ? Math.min(...values) : null;
        }
        case 'iif':
        case 'if': {
            const condition = evalArg(args[0]);
            return condition ? evalArg(args[1]) : evalArg(args[2]);
        }

        // ========== Type Casting ==========
        case 'cast': {
            const value = evalArg(args[0]);
            const targetType = String(args[1]?.value || args[1] || '').toUpperCase();
            if (value == null) return null;
            switch (targetType) {
                case 'INTEGER': case 'INT': case 'BIGINT': return Math.trunc(Number(value));
                case 'REAL': case 'FLOAT': case 'DOUBLE': return Number(value);
                case 'TEXT': case 'VARCHAR': case 'STRING': return String(value);
                case 'BOOLEAN': case 'BOOL': return Boolean(value);
                default: return value;
            }
        }

        // ========== Date/Time Functions ==========
        case 'now':
        case 'current_timestamp':
            return new Date().toISOString();
        case 'current_date':
            return new Date().toISOString().split('T')[0];
        case 'current_time':
            return new Date().toISOString().split('T')[1].split('.')[0];
        case 'date': {
            const val = evalArg(args[0]);
            if (!val) return null;
            const d = new Date(val);
            return isNaN(d.getTime()) ? null : d.toISOString().split('T')[0];
        }
        case 'time': {
            const val = evalArg(args[0]);
            if (!val) return null;
            const d = new Date(val);
            return isNaN(d.getTime()) ? null : d.toISOString().split('T')[1].split('.')[0];
        }
        case 'strftime': {
            const format = String(evalArg(args[0]) ?? '');
            const dateVal = evalArg(args[1]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            // Use UTC methods for consistency with date-only strings
            return format
                .replace(/%Y/g, d.getUTCFullYear())
                .replace(/%m/g, String(d.getUTCMonth() + 1).padStart(2, '0'))
                .replace(/%d/g, String(d.getUTCDate()).padStart(2, '0'))
                .replace(/%H/g, String(d.getUTCHours()).padStart(2, '0'))
                .replace(/%M/g, String(d.getUTCMinutes()).padStart(2, '0'))
                .replace(/%S/g, String(d.getUTCSeconds()).padStart(2, '0'))
                .replace(/%w/g, d.getUTCDay())
                .replace(/%j/g, Math.floor((d - new Date(Date.UTC(d.getUTCFullYear(), 0, 0))) / 86400000));
        }
        case 'date_diff': {
            const unit = String(evalArg(args[0]) ?? 'day').toLowerCase();
            const d1 = new Date(evalArg(args[1]));
            const d2 = new Date(evalArg(args[2]));
            if (isNaN(d1.getTime()) || isNaN(d2.getTime())) return null;
            const diffMs = d2.getTime() - d1.getTime();
            switch (unit) {
                case 'second': case 'seconds': return Math.floor(diffMs / 1000);
                case 'minute': case 'minutes': return Math.floor(diffMs / 60000);
                case 'hour': case 'hours': return Math.floor(diffMs / 3600000);
                case 'day': case 'days': return Math.floor(diffMs / 86400000);
                case 'week': case 'weeks': return Math.floor(diffMs / 604800000);
                case 'month': case 'months':
                    return (d2.getFullYear() - d1.getFullYear()) * 12 + (d2.getMonth() - d1.getMonth());
                case 'year': case 'years':
                    return d2.getFullYear() - d1.getFullYear();
                default: return Math.floor(diffMs / 86400000);
            }
        }
        case 'date_add':
        case 'date_sub': {
            const dateVal = evalArg(args[0]);
            const amount = evalArg(args[1]) ?? 0;
            const unit = String(evalArg(args[2]) ?? 'day').toLowerCase();
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            const sign = func === 'date_add' ? 1 : -1;
            switch (unit) {
                case 'second': case 'seconds': d.setSeconds(d.getSeconds() + sign * amount); break;
                case 'minute': case 'minutes': d.setMinutes(d.getMinutes() + sign * amount); break;
                case 'hour': case 'hours': d.setHours(d.getHours() + sign * amount); break;
                case 'day': case 'days': d.setDate(d.getDate() + sign * amount); break;
                case 'week': case 'weeks': d.setDate(d.getDate() + sign * amount * 7); break;
                case 'month': case 'months': d.setMonth(d.getMonth() + sign * amount); break;
                case 'year': case 'years': d.setFullYear(d.getFullYear() + sign * amount); break;
            }
            return d.toISOString();
        }
        case 'extract': {
            const unit = String(evalArg(args[0]) ?? '').toUpperCase();
            const dateVal = evalArg(args[1]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            switch (unit) {
                case 'YEAR': return d.getFullYear();
                case 'MONTH': return d.getMonth() + 1;
                case 'DAY': return d.getDate();
                case 'HOUR': return d.getHours();
                case 'MINUTE': return d.getMinutes();
                case 'SECOND': return d.getSeconds();
                case 'DOW': case 'DAYOFWEEK': return d.getDay();
                case 'DOY': case 'DAYOFYEAR':
                    return Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000);
                default: return null;
            }
        }
        // Shorthand date extractors (use UTC to avoid timezone issues with date-only strings)
        case 'year': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCFullYear();
        }
        case 'month': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCMonth() + 1;
        }
        case 'day': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCDate();
        }
        case 'hour': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCHours();
        }
        case 'minute': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCMinutes();
        }
        case 'second': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCSeconds();
        }

        // ========== Additional String Functions ==========
        case 'split': {
            const str = String(evalArg(args[0]) ?? '');
            const delimiter = String(evalArg(args[1]) ?? ',');
            return str.split(delimiter);
        }
        case 'left': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.substring(0, n);
        }
        case 'right': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.substring(Math.max(0, str.length - n));
        }
        case 'lpad': {
            const str = String(evalArg(args[0]) ?? '');
            const len = evalArg(args[1]) ?? 0;
            const pad = String(evalArg(args[2]) ?? ' ');
            return str.padStart(len, pad);
        }
        case 'rpad': {
            const str = String(evalArg(args[0]) ?? '');
            const len = evalArg(args[1]) ?? 0;
            const pad = String(evalArg(args[2]) ?? ' ');
            return str.padEnd(len, pad);
        }
        case 'position':
        case 'instr': {
            const str = String(evalArg(args[0]) ?? '');
            const substr = String(evalArg(args[1]) ?? '');
            const pos = str.indexOf(substr);
            return pos === -1 ? 0 : pos + 1; // SQL uses 1-based indexing
        }
        case 'repeat': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.repeat(Math.max(0, n));
        }
        case 'reverse': {
            const str = String(evalArg(args[0]) ?? '');
            return str.split('').reverse().join('');
        }

        // ========== REGEXP Functions ==========
        case 'regexp_matches': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const flags = args[2] ? String(evalArg(args[2])) : '';
            try {
                return new RegExp(pattern, flags).test(str) ? 1 : 0;
            } catch (e) {
                return 0;
            }
        }
        case 'regexp_replace': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const replacement = String(evalArg(args[2]) ?? '');
            const flags = args[3] ? String(evalArg(args[3])) : 'g';
            try {
                return str.replace(new RegExp(pattern, flags), replacement);
            } catch (e) {
                return str;
            }
        }
        case 'regexp_extract':
        case 'regexp_substr': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const groupIndex = args[2] ? parseInt(evalArg(args[2]), 10) : 0;
            try {
                const match = str.match(new RegExp(pattern));
                return match ? (match[groupIndex] ?? null) : null;
            } catch (e) {
                return null;
            }
        }
        case 'regexp_count': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const flags = (args[2] ? String(evalArg(args[2])) : '') + 'g';
            try {
                const matches = str.match(new RegExp(pattern, flags));
                return matches ? matches.length : 0;
            } catch (e) {
                return 0;
            }
        }
        case 'regexp_split': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            try {
                return JSON.stringify(str.split(new RegExp(pattern)));
            } catch (e) {
                return JSON.stringify([str]);
            }
        }

        // ========== JSON Functions ==========
        case 'json_extract':
        case 'json_value': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            const path = String(evalArg(args[1]) ?? '$');
            try {
                const obj = JSON.parse(jsonStr);
                return navigateJsonPath(obj, path);
            } catch (e) {
                return null;
            }
        }
        case 'json_object': {
            const result = {};
            for (let i = 0; i < args.length; i += 2) {
                const key = String(evalArg(args[i]) ?? '');
                const value = evalArg(args[i + 1]);
                result[key] = value;
            }
            return JSON.stringify(result);
        }
        case 'json_array': {
            const result = args.map(evalArg);
            return JSON.stringify(result);
        }
        case 'json_keys': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            try {
                const obj = JSON.parse(jsonStr);
                if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
                    return JSON.stringify(Object.keys(obj));
                }
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_length': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            try {
                const obj = JSON.parse(jsonStr);
                if (Array.isArray(obj)) return obj.length;
                if (typeof obj === 'object' && obj !== null) return Object.keys(obj).length;
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_type': {
            const jsonStr = String(evalArg(args[0]) ?? 'null');
            try {
                const obj = JSON.parse(jsonStr);
                if (obj === null) return 'NULL';
                if (Array.isArray(obj)) return 'ARRAY';
                if (typeof obj === 'object') return 'OBJECT';
                if (typeof obj === 'string') return 'STRING';
                if (typeof obj === 'number') return 'NUMBER';
                if (typeof obj === 'boolean') return 'BOOLEAN';
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_valid': {
            const jsonStr = String(evalArg(args[0]) ?? '');
            try {
                JSON.parse(jsonStr);
                return 1;
            } catch (e) {
                return 0;
            }
        }

        // ========== Array Functions ==========
        case 'array_length': {
            const arr = evalArg(args[0]);
            return Array.isArray(arr) ? arr.length : null;
        }
        case 'array_contains': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return arr.includes(value) ? 1 : 0;
        }
        case 'array_position': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            const idx = arr.indexOf(value);
            return idx === -1 ? null : idx + 1; // SQL 1-based indexing
        }
        case 'array_append': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return [...arr, value];
        }
        case 'array_remove': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return arr.filter(el => el !== value);
        }
        case 'array_slice': {
            const arr = evalArg(args[0]);
            const start = (evalArg(args[1]) ?? 1) - 1; // SQL 1-based to JS 0-based
            const end = args[2] ? evalArg(args[2]) - 1 : arr?.length; // SQL 1-based to JS 0-based (exclusive)
            if (!Array.isArray(arr)) return null;
            return arr.slice(start, end);
        }
        case 'array_concat': {
            const arr1 = evalArg(args[0]);
            const arr2 = evalArg(args[1]);
            if (!Array.isArray(arr1) || !Array.isArray(arr2)) return null;
            return [...arr1, ...arr2];
        }
        case 'unnest': {
            // UNNEST typically expands array to rows - in scalar context, return first element
            const arr = evalArg(args[0]);
            return Array.isArray(arr) && arr.length > 0 ? arr[0] : null;
        }

        // ========== UUID Functions ==========
        case 'uuid':
        case 'gen_random_uuid': {
            // Generate UUID v4
            if (typeof crypto !== 'undefined' && crypto.randomUUID) {
                return crypto.randomUUID();
            }
            // Fallback for environments without crypto.randomUUID
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        }
        case 'uuid_string': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            return String(val);
        }
        case 'is_uuid': {
            const val = evalArg(args[0]);
            if (val == null) return 0;
            const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
            return uuidRegex.test(String(val)) ? 1 : 0;
        }

        // ========== Binary/Bit Functions ==========
        case 'bit_count': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            let n = val | 0;
            let count = 0;
            // Handle negative numbers by using unsigned right shift
            n = n >>> 0;
            while (n) {
                count += n & 1;
                n >>>= 1;
            }
            return count;
        }
        case 'hex': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            if (typeof val === 'number') {
                return (val >>> 0).toString(16).toUpperCase();
            }
            // String to hex
            return String(val).split('').map(c => c.charCodeAt(0).toString(16).padStart(2, '0')).join('').toUpperCase();
        }
        case 'unhex': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            const hex = String(val);
            let result = '';
            for (let i = 0; i < hex.length; i += 2) {
                result += String.fromCharCode(parseInt(hex.substr(i, 2), 16));
            }
            return result;
        }
        case 'encode': {
            const val = evalArg(args[0]);
            const encoding = String(evalArg(args[1]) ?? 'base64').toLowerCase();
            if (val == null) return null;
            if (encoding === 'base64') {
                return btoa(String(val));
            }
            if (encoding === 'hex') {
                return String(val).split('').map(c => c.charCodeAt(0).toString(16).padStart(2, '0')).join('');
            }
            return val;
        }
        case 'decode': {
            const val = evalArg(args[0]);
            const encoding = String(evalArg(args[1]) ?? 'base64').toLowerCase();
            if (val == null) return null;
            if (encoding === 'base64') {
                try {
                    return atob(String(val));
                } catch (e) {
                    return null;
                }
            }
            if (encoding === 'hex') {
                const hex = String(val);
                let result = '';
                for (let i = 0; i < hex.length; i += 2) {
                    result += String.fromCharCode(parseInt(hex.substr(i, 2), 16));
                }
                return result;
            }
            return val;
        }

        // GROUPING function for ROLLUP/CUBE/GROUPING SETS
        case 'grouping': {
            // GROUPING(column) returns 1 if column is a super-aggregate (rolled up), 0 otherwise
            const arg = args[0];
            if (arg && arg.type === 'column') {
                const colName = typeof arg.value === 'string' ? arg.value : arg.value.column;
                return row[`__grouping_${colName}`] ?? 0;
            }
            return 0;
        }

        default:
            return null;
    }
}

// JSON path navigation helper (supports $.key.subkey and $.array[0])
function parseJsonPath(path) {
    if (!path || !path.startsWith('$')) return [];
    const segments = [];
    let remaining = path.substring(1);
    while (remaining.length > 0) {
        if (remaining.startsWith('.')) {
            remaining = remaining.substring(1);
            const match = remaining.match(/^([a-zA-Z_][a-zA-Z0-9_]*)/);
            if (match) {
                segments.push({ type: 'key', value: match[1] });
                remaining = remaining.substring(match[1].length);
            }
        } else if (remaining.startsWith('[')) {
            const endBracket = remaining.indexOf(']');
            if (endBracket === -1) break;
            const content = remaining.substring(1, endBracket);
            if (/^\d+$/.test(content)) {
                segments.push({ type: 'index', value: parseInt(content, 10) });
            } else if (content.startsWith("'") || content.startsWith('"')) {
                segments.push({ type: 'key', value: content.slice(1, -1) });
            }
            remaining = remaining.substring(endBracket + 1);
        } else {
            break;
        }
    }
    return segments;
}

function navigateJsonPath(obj, path) {
    let current = obj;
    for (const seg of parseJsonPath(path)) {
        if (current == null) return null;
        current = seg.type === 'key' ? current[seg.value] :
                  (Array.isArray(current) ? current[seg.value] : null);
    }
    return typeof current === 'object' ? JSON.stringify(current) : current;
}

// Evaluate scalar subquery for a single outer row (correlated subquery support)
async function evaluateScalarSubquery(subquery, outerRow, db, tableAliases) {
    // Bind outer row references into the subquery's WHERE clause
    const boundSubquery = JSON.parse(JSON.stringify(subquery)); // Deep clone
    if (boundSubquery.where) {
        bindOuterReferences(boundSubquery.where, outerRow, tableAliases);
    }

    const result = await executeAST(db, boundSubquery);
    if (result.rows.length === 0) return null;
    if (result.rows.length > 1) throw new Error('Scalar subquery returned more than one row');

    // Return the first (and only) column value of the first row
    const keys = Object.keys(result.rows[0]);
    return keys.length > 0 ? result.rows[0][keys[0]] : null;
}

// Bind outer row references in WHERE clause for correlated subqueries
function bindOuterReferences(where, outerRow, tableAliases) {
    if (!where) return;

    if (where.op === 'AND' || where.op === 'OR') {
        bindOuterReferences(where.left, outerRow, tableAliases);
        bindOuterReferences(where.right, outerRow, tableAliases);
        return;
    }

    // Check if the value references an outer table column
    if (where.value && typeof where.value === 'object' && where.value.table) {
        const val = getColumnValue(outerRow, where.value, tableAliases);
        if (val !== undefined) {
            where.value = val; // Replace with actual value
        }
    }

    // Also check the column side for cases like outer.col = inner.col
    if (where.column && typeof where.column === 'object' && where.column.table) {
        const val = getColumnValue(outerRow, where.column, tableAliases);
        if (val !== undefined && where.value !== undefined) {
            // Swap: column becomes the inner column, value becomes the outer value
            where.column = where.value;
            where.value = val;
        }
    }
}

// Evaluate CASE WHEN expression for a single row
function evaluateCaseExpr(caseExpr, row, tableAliases = {}) {
    const evalArg = (arg) => {
        if (!arg) return null;
        if (arg.type === 'literal') return arg.value;
        if (arg.type === 'column') return getColumnValue(row, arg.value, tableAliases);
        if (arg.type === 'function') return evaluateScalarFunction(arg.func, arg.args, row, tableAliases);
        if (arg.type === 'array_literal') return arg.elements.map(el => evalArg(el));
        if (arg.type === 'subscript') {
            const arr = evalArg(arg.array);
            const idx = evalArg(arg.index);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null;
        }
        if (arg.type === 'arithmetic') {
            if (arg.op === 'unary-') return -evalArg(arg.operand);
            if (arg.op === 'unary~') return ~(evalArg(arg.operand) | 0);
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        }
        if (arg.type === 'comparison') {
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '=': return left === right;
                case '!=': case '<>': return left !== right;
                case '<': return left < right;
                case '<=': return left <= right;
                case '>': return left > right;
                case '>=': return left >= right;
                default: return false;
            }
        }
        return null;
    };

    // Simple CASE: CASE expr WHEN value THEN result
    if (caseExpr.caseExpr) {
        const caseVal = evalArg(caseExpr.caseExpr);
        for (const branch of caseExpr.branches) {
            const whenVal = evalArg(branch.condition);
            if (caseVal === whenVal) {
                return evalArg(branch.result);
            }
        }
    } else {
        // Searched CASE: CASE WHEN condition THEN result
        for (const branch of caseExpr.branches) {
            const cond = evalArg(branch.condition);
            if (cond) {
                return evalArg(branch.result);
            }
        }
    }

    // ELSE clause
    return caseExpr.elseResult ? evalArg(caseExpr.elseResult) : null;
}

// Compute window functions for all rows
function computeWindowFunctions(rows, windowCols, tableAliases = {}) {
    if (rows.length === 0) return rows;

    for (const col of windowCols) {
        const alias = col.alias || `${col.func}(...)`;
        const windowKey = `__window_${alias}`;
        const { partitionBy, orderBy } = col.over;

        // Group rows into partitions
        const partitions = new Map();
        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            const partKey = partitionBy
                ? partitionBy.map(p => {
                    const colName = typeof p === 'string' ? p : p.column;
                    return String(row[colName]);
                }).join('|')
                : '__all__';
            if (!partitions.has(partKey)) {
                partitions.set(partKey, []);
            }
            partitions.get(partKey).push({ index: i, row });
        }

        // Process each partition
        for (const [, partRows] of partitions) {
            // Sort partition by ORDER BY if specified
            if (orderBy && orderBy.length > 0) {
                partRows.sort((a, b) => {
                    for (const order of orderBy) {
                        const colName = typeof order.column === 'string' ? order.column : order.column.column;
                        const aVal = a.row[colName];
                        const bVal = b.row[colName];
                        if (aVal < bVal) return order.desc ? 1 : -1;
                        if (aVal > bVal) return order.desc ? -1 : 1;
                    }
                    return 0;
                });
            }

            // Helper to get frame bounds
            const getFrameBounds = (i, frame) => {
                if (!frame) {
                    // Default: if ORDER BY exists, UNBOUNDED PRECEDING to CURRENT ROW
                    // Otherwise, entire partition
                    return orderBy && orderBy.length > 0 ? [0, i] : [0, partRows.length - 1];
                }
                let start = 0, end = partRows.length - 1;
                // Calculate start
                if (frame.start.type === 'unbounded' && frame.start.direction === 'preceding') {
                    start = 0;
                } else if (frame.start.type === 'current') {
                    start = i;
                } else if (frame.start.type === 'offset') {
                    start = frame.start.direction === 'preceding' ? i - frame.start.value : i + frame.start.value;
                }
                // Calculate end
                if (frame.end.type === 'unbounded' && frame.end.direction === 'following') {
                    end = partRows.length - 1;
                } else if (frame.end.type === 'current') {
                    end = i;
                } else if (frame.end.type === 'offset') {
                    end = frame.end.direction === 'preceding' ? i - frame.end.value : i + frame.end.value;
                }
                return [Math.max(0, start), Math.min(partRows.length - 1, end)];
            };

            // Compute window function for each row in partition
            for (let i = 0; i < partRows.length; i++) {
                const { index, row } = partRows[i];
                let value;
                const frame = col.over.frame;

                switch (col.func) {
                    case 'row_number':
                        value = i + 1;
                        break;
                    case 'rank': {
                        // Same rank for same ORDER BY values
                        if (i === 0) {
                            value = 1;
                        } else {
                            const prevRow = partRows[i - 1].row;
                            const sameAsPrev = orderBy ? orderBy.every(order => {
                                const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                return row[colName] === prevRow[colName];
                            }) : false;
                            value = sameAsPrev ? rows[partRows[i - 1].index][windowKey] : i + 1;
                        }
                        break;
                    }
                    case 'dense_rank': {
                        if (i === 0) {
                            value = 1;
                        } else {
                            const prevRow = partRows[i - 1].row;
                            const sameAsPrev = orderBy ? orderBy.every(order => {
                                const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                return row[colName] === prevRow[colName];
                            }) : false;
                            value = sameAsPrev ? rows[partRows[i - 1].index][windowKey] : rows[partRows[i - 1].index][windowKey] + 1;
                        }
                        break;
                    }
                    case 'sum': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        value = windowRows.reduce((acc, p) => {
                            const v = p.row[argCol];
                            return acc + (typeof v === 'number' ? v : 0);
                        }, 0);
                        break;
                    }
                    case 'count': {
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        value = frameEnd - frameStart + 1;
                        break;
                    }
                    case 'avg': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => typeof v === 'number');
                        value = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
                        break;
                    }
                    case 'min': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => v != null);
                        value = vals.length > 0 ? Math.min(...vals) : null;
                        break;
                    }
                    case 'max': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => v != null);
                        value = vals.length > 0 ? Math.max(...vals) : null;
                        break;
                    }
                    case 'lag': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = i > 0 ? partRows[i - 1].row[argCol] : null;
                        break;
                    }
                    case 'lead': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = i < partRows.length - 1 ? partRows[i + 1].row[argCol] : null;
                        break;
                    }
                    case 'ntile': {
                        // NTILE(n) divides partition into n buckets
                        const n = col.arg || 1;
                        const bucketSize = Math.ceil(partRows.length / n);
                        value = Math.min(Math.floor(i / bucketSize) + 1, n);
                        break;
                    }
                    case 'percent_rank': {
                        // (rank - 1) / (partition_size - 1)
                        if (partRows.length <= 1) {
                            value = 0;
                        } else {
                            // Calculate rank first
                            let rank = 1;
                            if (i > 0 && orderBy) {
                                for (let j = 0; j < i; j++) {
                                    const jRow = partRows[j].row;
                                    const isSame = orderBy.every(order => {
                                        const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                        return row[colName] === jRow[colName];
                                    });
                                    if (!isSame) rank = j + 2;
                                }
                            }
                            value = (rank - 1) / (partRows.length - 1);
                        }
                        break;
                    }
                    case 'cume_dist': {
                        // count of rows <= current / partition_size
                        let count = i + 1;
                        if (orderBy) {
                            // Count rows with same or lower order value
                            for (let j = i + 1; j < partRows.length; j++) {
                                const jRow = partRows[j].row;
                                const isSame = orderBy.every(order => {
                                    const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                    return row[colName] === jRow[colName];
                                });
                                if (isSame) count++;
                                else break;
                            }
                        }
                        value = count / partRows.length;
                        break;
                    }
                    case 'first_value': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = partRows[0].row[argCol];
                        break;
                    }
                    case 'last_value': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        // Default frame is UNBOUNDED PRECEDING to CURRENT ROW, so last_value = current row
                        // With explicit UNBOUNDED FOLLOWING, it's partition last
                        const frame = col.over.frame;
                        if (frame && frame.end.type === 'unbounded' && frame.end.direction === 'following') {
                            value = partRows[partRows.length - 1].row[argCol];
                        } else {
                            // Default: current row
                            value = row[argCol];
                        }
                        break;
                    }
                    case 'nth_value': {
                        // NTH_VALUE(col, n) - returns nth value in partition
                        const argCol = col.args ? (typeof col.args[0] === 'string' ? col.args[0] : col.args[0]?.column) : (typeof col.arg === 'string' ? col.arg : col.arg?.column);
                        const n = col.args ? col.args[1] : 1;
                        value = n > 0 && n <= partRows.length ? partRows[n - 1].row[argCol] : null;
                        break;
                    }
                    default:
                        value = null;
                }

                rows[index][windowKey] = value;
            }
        }
    }

    return rows;
}

// Evaluate HAVING clause (similar to WHERE but works on aggregated values)
function evalHaving(having, row) {
    if (!having) return true;

    switch (having.op) {
        case 'AND':
            return evalHaving(having.left, row) && evalHaving(having.right, row);
        case 'OR':
            return evalHaving(having.left, row) || evalHaving(having.right, row);
        default: {
            // For aggregate comparisons, the column is already in the row
            let val;
            if (having.column && having.column.type === 'aggregate') {
                const aggName = `${having.column.func}(${having.column.arg === '*' ? '*' : (typeof having.column.arg === 'string' ? having.column.arg : having.column.arg.column)})`;
                val = row[aggName];
            } else {
                const colName = typeof having.column === 'string' ? having.column : having.column.column;
                val = row[colName];
            }

            switch (having.op) {
                case '=': return val === having.value;
                case '!=':
                case '<>': return val !== having.value;
                case '<': return val < having.value;
                case '<=': return val <= having.value;
                case '>': return val > having.value;
                case '>=': return val >= having.value;
                default: return true;
            }
        }
    }
}

// Extract NEAR condition from WHERE expression
function extractNearCondition(expr) {
    if (!expr) return null;
    if (expr.op === 'NEAR') {
        return expr;
    }
    if (expr.op === 'AND' || expr.op === 'OR') {
        const leftNear = extractNearCondition(expr.left);
        if (leftNear) return leftNear;
        return extractNearCondition(expr.right);
    }
    return null;
}

// Remove NEAR condition from expression, returning remaining conditions
function removeNearCondition(expr) {
    if (!expr) return null;
    if (expr.op === 'NEAR') return null;
    if (expr.op === 'AND' || expr.op === 'OR') {
        const left = removeNearCondition(expr.left);
        const right = removeNearCondition(expr.right);
        if (!left && !right) return null;
        if (!left) return right;
        if (!right) return left;
        return { op: expr.op, left, right };
    }
    return expr;
}

// Cosine similarity between two vectors
function cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// Execute NEAR vector search
async function executeNearSearch(rows, nearCondition, limit) {
    const column = typeof nearCondition.column === 'string' ? nearCondition.column : nearCondition.column.column;
    const text = nearCondition.text;
    const topK = nearCondition.topK || limit || 10;

    // Check if gpuTransformer is available and has a loaded model
    const gpuTransformer = getGPUTransformerState();
    if (!gpuTransformer) {
        throw new Error('NEAR requires a text encoder model. Load a model first with store.loadModel()');
    }

    // Generate query embedding
    let queryVec;
    try {
        // Try to get any loaded model
        const models = gpuTransformer.getLoadedModels?.() || [];
        if (models.length === 0) {
            throw new Error('No text encoder model loaded');
        }
        queryVec = await gpuTransformer.encodeText(text, models[0]);
    } catch (e) {
        throw new Error(`NEAR failed to encode query: ${e.message}`);
    }

    // Score each row
    const scored = [];
    const embeddingCache = getEmbeddingCache();
    for (const row of rows) {
        const colValue = row[column];

        // If column is already a vector (array of numbers), use it directly
        if (Array.isArray(colValue) && typeof colValue[0] === 'number') {
            const score = cosineSimilarity(queryVec, colValue);
            scored.push({ row, score });
        }
        // If column is text, we need to embed it (expensive)
        else if (typeof colValue === 'string') {
            const cacheKey = `sql:${column}:${colValue}`;
            let itemVec = embeddingCache.get(cacheKey);
            if (!itemVec) {
                const models = gpuTransformer.getLoadedModels?.() || [];
                itemVec = await gpuTransformer.encodeText(colValue, models[0]);
                embeddingCache.set(cacheKey, itemVec);
            }
            const score = cosineSimilarity(queryVec, itemVec);
            scored.push({ row, score });
        }
    }

    // Sort by score descending and take top K
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => ({ ...s.row, _score: s.score }));
}

// Generate query plan from AST for EXPLAIN
function generateQueryPlan(ast) {
    const type = ast.type?.toUpperCase() || 'SELECT';

    // Handle JOINs - they become HASH_JOIN operations
    if (type === 'SELECT' && ast.joins && ast.joins.length > 0) {
        const children = [];

        // Main table scan
        const mainTable = ast.tables?.[0];
        children.push({
            operation: 'SELECT',
            table: typeof mainTable === 'string' ? mainTable : (mainTable?.name || mainTable?.alias || 'unknown'),
            access: 'FULL_SCAN'
        });

        // Join tables
        for (const join of ast.joins) {
            const joinTable = join.table;
            children.push({
                operation: 'SELECT',
                table: typeof joinTable === 'string' ? joinTable : (joinTable?.name || joinTable?.alias || 'unknown'),
                access: 'FULL_SCAN'
            });
        }

        return {
            operation: 'HASH_JOIN',
            joinType: ast.joins[0].type || 'INNER',
            children
        };
    }

    // Standard SELECT/UPDATE/DELETE/INSERT
    const plan = {
        operation: type,
        table: getTableName(ast),
        access: 'FULL_SCAN'
    };

    // Detect optimizations
    const optimizations = [];

    if (ast.where) {
        optimizations.push('PREDICATE_PUSHDOWN');
        plan.filter = stringifyWhereClause(ast.where);
    }

    if (ast.groupBy) {
        optimizations.push('AGGREGATE');
    }

    if (ast.orderBy && ast.orderBy.length > 0) {
        optimizations.push('SORT');
    }

    if (ast.limit !== undefined) {
        optimizations.push('LIMIT');
    }

    if (optimizations.length > 0) {
        plan.optimizations = optimizations;
    }

    return plan;
}

function getTableName(ast) {
    if (ast.table) return ast.table;
    if (ast.tables && ast.tables.length > 0) {
        const t = ast.tables[0];
        return typeof t === 'string' ? t : (t.name || t.alias || 'unknown');
    }
    return 'unknown';
}

function stringifyWhereClause(where) {
    if (!where) return '';
    if (where.type === 'comparison') {
        const left = stringifyExpr(where.left);
        const right = stringifyExpr(where.right);
        return `${left} ${where.op} ${right}`;
    }
    if (where.type === 'logical') {
        const left = stringifyWhereClause(where.left);
        const right = stringifyWhereClause(where.right);
        return `(${left} ${where.op} ${right})`;
    }
    return JSON.stringify(where);
}

function stringifyExpr(expr) {
    if (!expr) return 'null';
    if (expr.type === 'literal') return String(expr.value);
    if (expr.type === 'column') {
        return typeof expr.value === 'string' ? expr.value : expr.value.column;
    }
    return JSON.stringify(expr);
}

async function executeSQL(db, sql) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    const ast = parser.parse();
    return executeAST(db, ast);
}

async function executeAST(db, ast) {
    const type = ast.type.toUpperCase();

    switch (type) {
        case 'CREATE_TABLE':
            return db.createTable(ast.table, ast.columns, ast.ifNotExists);

        case 'DROP_TABLE':
            return db.dropTable(ast.table, ast.ifExists);

        case 'INSERT': {
            let rows = ast.rows || [];

            // Handle INSERT...SELECT
            if (ast.select) {
                const selectResult = await executeAST(db, ast.select);
                rows = selectResult.rows;

                // Map select columns to insert columns if specified
                if (ast.columns && rows.length > 0) {
                    const selectCols = Object.keys(rows[0]);
                    rows = rows.map(row => {
                        const newRow = {};
                        ast.columns.forEach((col, i) => {
                            newRow[col] = row[selectCols[i]];
                        });
                        return newRow;
                    });
                }
            } else if (rows.length > 0 && Array.isArray(rows[0])) {
                // If rows are arrays (no column names in INSERT), convert to objects using table schema
                const tableState = db.tables.get(ast.table);
                if (tableState && tableState.schema) {
                    rows = rows.map(valueArray => {
                        const row = {};
                        tableState.schema.forEach((col, i) => {
                            row[col.name] = valueArray[i];
                        });
                        return row;
                    });
                }
            }

            // Handle ON CONFLICT (UPSERT)
            if (ast.onConflict) {
                const tableState = db.tables.get(ast.table);
                const existingRows = await db.select(ast.table, {});
                const conflictCols = ast.onConflict.columns ||
                    tableState?.schema?.filter(c => c.primaryKey).map(c => c.name) || ['id'];

                const insertedRows = [];
                const updatedRows = [];

                for (const row of rows) {
                    const existingRow = existingRows.find(existing =>
                        conflictCols.every(col => existing[col] === row[col])
                    );

                    if (existingRow) {
                        if (ast.onConflict.action === 'update') {
                            // Evaluate update expressions
                            const updates = {};
                            for (const [col, expr] of Object.entries(ast.onConflict.updates)) {
                                updates[col] = evaluateArithmeticExpr(expr, existingRow, {}, row);
                            }
                            // Apply updates to existing row
                            await db.updateWithExpr(ast.table, ast.onConflict.updates,
                                (r) => conflictCols.every(col => r[col] === row[col]),
                                (expr, r) => evaluateArithmeticExpr(expr, r, {}, row));
                        }
                        // 'nothing': skip insertion
                    } else {
                        insertedRows.push(row);
                    }
                }
                if (insertedRows.length > 0) {
                    return db.insert(ast.table, insertedRows);
                }
                return { success: true };
            }

            return db.insert(ast.table, rows);
        }

        case 'DELETE': {
            // Handle DELETE with USING clause (JOIN-based delete)
            if (ast.using) {
                const mainRows = await db.select(ast.table, {});
                const tableAliases = { [ast.alias || ast.table]: ast.table };

                // Build joined rows
                let joinedRows = mainRows.map((r) => ({ [ast.alias || ast.table]: { ...r } }));
                for (const t of ast.using) {
                    const rightRows = await db.select(t.name, {});
                    tableAliases[t.alias || t.name] = t.name;
                    const newJoined = [];
                    for (const left of joinedRows) {
                        for (const right of rightRows) {
                            newJoined.push({ ...left, [t.alias || t.name]: right });
                        }
                    }
                    joinedRows = newJoined;
                }

                // Filter with WHERE to find matching rows
                if (ast.where) {
                    joinedRows = joinedRows.filter(jr => {
                        const flatRow = flattenJoinedRow(jr);
                        return evalWhere(ast.where, flatRow);
                    });
                }

                // Collect main table rows to delete
                const rowsToDelete = joinedRows.map(jr => jr[ast.alias || ast.table]);
                const tableSchema = db.tables.get(ast.table)?.schema || [];

                // Delete matching rows by comparing all columns
                return db.delete(ast.table, (row) => {
                    return rowsToDelete.some(delRow =>
                        tableSchema.every(col => row[col.name] === delRow[col.name])
                    );
                });
            }

            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;
            return db.delete(ast.table, predicate);
        }

        case 'UPDATE': {
            // Handle UPDATE with FROM clause (JOIN-based update)
            if (ast.from) {
                const mainRows = await db.select(ast.table, {});
                const tableAliases = { [ast.alias || ast.table]: ast.table };

                // Build joined rows
                let joinedRows = mainRows.map((r) => ({ [ast.alias || ast.table]: { ...r } }));
                for (const t of ast.from) {
                    const rightRows = await db.select(t.name, {});
                    tableAliases[t.alias || t.name] = t.name;
                    const newJoined = [];
                    for (const left of joinedRows) {
                        for (const right of rightRows) {
                            newJoined.push({ ...left, [t.alias || t.name]: right });
                        }
                    }
                    joinedRows = newJoined;
                }

                // Filter with WHERE
                if (ast.where) {
                    joinedRows = joinedRows.filter(jr => {
                        const flatRow = flattenJoinedRow(jr);
                        return evalWhere(ast.where, flatRow);
                    });
                }

                // Collect matched rows and their update contexts
                const matchedContexts = [];
                for (const jr of joinedRows) {
                    const mainRow = jr[ast.alias || ast.table];
                    matchedContexts.push({
                        mainRow,
                        context: flattenJoinedRow(jr)
                    });
                }

                // Use updateWithExpr - check if row matches any in matchedContexts
                const seen = new Set();
                return db.updateWithExpr(ast.table, ast.updates,
                    (row) => {
                        // Check if this row matches any of the main rows
                        const tableSchema = db.tables.get(ast.table)?.schema || [];
                        for (const m of matchedContexts) {
                            const matches = tableSchema.every(col =>
                                row[col.name] === m.mainRow[col.name]
                            );
                            if (matches && !seen.has(JSON.stringify(m.mainRow))) {
                                seen.add(JSON.stringify(m.mainRow));
                                row.__updateContext = m.context;
                                return true;
                            }
                        }
                        return false;
                    },
                    (expr, row) => {
                        const context = row.__updateContext || row;
                        delete row.__updateContext;
                        return evaluateArithmeticExpr(expr, context, tableAliases);
                    });
            }

            // Simple UPDATE (no FROM) - use updateWithExpr which handles expressions
            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;

            return db.updateWithExpr(ast.table, ast.updates, predicate, (expr, row) => evaluateArithmeticExpr(expr, row, {}));
        }

        case 'SELECT': {
            // Build table alias mapping and handle derived tables (subqueries)
            const tableAliases = {};
            const derivedTables = new Map();

            if (ast.tables) {
                for (const t of ast.tables) {
                    if (t.type === 'subquery') {
                        // Execute subquery and store as derived table
                        const subResult = await executeAST(db, t.query);
                        derivedTables.set(t.alias, subResult.rows);
                        tableAliases[t.alias] = t.alias;
                    } else if (t.alias) {
                        tableAliases[t.alias] = t.name;
                    }
                }
            }
            if (ast.joins) {
                for (const j of ast.joins) {
                    if (j.table.type === 'subquery') {
                        const subResult = await executeAST(db, j.table.query);
                        derivedTables.set(j.table.alias, subResult.rows);
                        tableAliases[j.table.alias] = j.table.alias;
                    } else if (j.table.alias) {
                        tableAliases[j.table.alias] = j.table.name;
                    }
                }
            }

            // Fetch data from main table (or derived table)
            // SELECT without FROM (e.g., SELECT 1+1, SELECT JSON_OBJECT(...)) returns single row
            let rows;
            const mainTable = ast.tables?.[0];
            if (!mainTable) {
                // No FROM clause - create a single empty row for expression evaluation
                rows = [{}];
            } else if (mainTable.type === 'subquery') {
                rows = derivedTables.get(mainTable.alias) || [];
            } else {
                rows = await db.select(ast.table, {});
            }

            // Process JOINs
            if (ast.joins && ast.joins.length > 0) {
                // Add prefixes for the first table before any JOINs
                const firstTableName = ast.tables[0].alias || ast.tables[0].name;
                rows = rows.map(row => {
                    const prefixed = { ...row };
                    for (const key of Object.keys(row)) {
                        prefixed[`${firstTableName}.${key}`] = row[key];
                    }
                    return prefixed;
                });

                for (const join of ast.joins) {
                    // Get right table rows (from database or derived table)
                    let rightRows;
                    if (join.table.type === 'subquery') {
                        rightRows = derivedTables.get(join.table.alias) || [];
                    } else {
                        rightRows = await db.select(join.table.name, {});
                    }
                    const newRows = [];
                    const matchedRightIndices = new Set();

                    // Get right table info for namespacing
                    const rightTableName = join.table.alias || join.table.name;

                    // CROSS JOIN: Cartesian product (no ON condition)
                    if (join.type === 'CROSS') {
                        for (const leftRow of rows) {
                            for (const rightRow of rightRows) {
                                const merged = { ...leftRow };
                                for (const key of Object.keys(rightRow)) {
                                    if (!(key in merged)) merged[key] = rightRow[key];
                                    merged[`${rightTableName}.${key}`] = rightRow[key];
                                }
                                newRows.push(merged);
                            }
                        }
                    } else {
                        // INNER, LEFT, RIGHT, FULL JOINs with ON condition
                        for (const leftRow of rows) {
                            let matched = false;
                            for (let ri = 0; ri < rightRows.length; ri++) {
                                const rightRow = rightRows[ri];
                                // Evaluate ON condition (supports compound AND/OR conditions)
                                if (evalJoinCondition(join.on, leftRow, rightRow, tableAliases)) {
                                    matched = true;
                                    matchedRightIndices.add(ri);
                                    // Merge rows - keep left row as-is, add right row with prefix
                                    const merged = { ...leftRow };
                                    for (const key of Object.keys(rightRow)) {
                                        if (!(key in merged)) merged[key] = rightRow[key];
                                        merged[`${rightTableName}.${key}`] = rightRow[key];
                                    }
                                    newRows.push(merged);
                                }
                            }
                            // LEFT JOIN or FULL OUTER JOIN: include left row even if no match
                            if (!matched && (join.type === 'LEFT' || join.type === 'FULL')) {
                                newRows.push({ ...leftRow });
                            }
                        }

                        // RIGHT JOIN or FULL OUTER JOIN: include unmatched right rows
                        if (join.type === 'RIGHT' || join.type === 'FULL') {
                            for (let ri = 0; ri < rightRows.length; ri++) {
                                if (!matchedRightIndices.has(ri)) {
                                    const merged = {};
                                    for (const key of Object.keys(rightRows[ri])) {
                                        merged[key] = rightRows[ri][key];
                                        merged[`${rightTableName}.${key}`] = rightRows[ri][key];
                                    }
                                    newRows.push(merged);
                                }
                            }
                        }
                    }
                    rows = newRows;
                }
            }

            // Pre-execute any subqueries in WHERE clause
            if (ast.where) {
                await preExecuteSubqueries(ast.where, db);
            }

            // Check for NEAR condition in WHERE clause
            const nearCondition = extractNearCondition(ast.where);
            if (nearCondition) {
                // Execute NEAR search first
                rows = await executeNearSearch(rows, nearCondition, ast.limit);
                // Apply remaining WHERE conditions
                const remainingWhere = removeNearCondition(ast.where);
                if (remainingWhere) {
                    rows = rows.filter(row => evalWhere(remainingWhere, row, tableAliases));
                }
            } else if (ast.where) {
                // Apply regular WHERE clause
                rows = rows.filter(row => evalWhere(ast.where, row, tableAliases));
            }

            // Apply GROUP BY with aggregations (supports ROLLUP/CUBE/GROUPING SETS)
            if (ast.groupBy && (Array.isArray(ast.groupBy) ? ast.groupBy.length > 0 : ast.groupBy.type)) {
                // Determine grouping sets based on type
                let groupingSets = [];
                let allColumns = [];

                if (Array.isArray(ast.groupBy)) {
                    // Standard GROUP BY: single grouping set with all columns
                    allColumns = ast.groupBy.map(col => typeof col === 'string' ? col : col.column);
                    groupingSets = [allColumns];
                } else if (ast.groupBy.type === 'ROLLUP') {
                    // ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
                    allColumns = ast.groupBy.columns.map(col => typeof col === 'string' ? col : col.column);
                    for (let i = allColumns.length; i >= 0; i--) {
                        groupingSets.push(allColumns.slice(0, i));
                    }
                } else if (ast.groupBy.type === 'CUBE') {
                    // CUBE(a, b) generates all 2^n combinations: (a,b), (a), (b), ()
                    allColumns = ast.groupBy.columns.map(col => typeof col === 'string' ? col : col.column);
                    const n = allColumns.length;
                    for (let mask = (1 << n) - 1; mask >= 0; mask--) {
                        const set = [];
                        for (let i = 0; i < n; i++) {
                            if (mask & (1 << i)) set.push(allColumns[i]);
                        }
                        groupingSets.push(set);
                    }
                } else if (ast.groupBy.type === 'GROUPING_SETS') {
                    // Explicit grouping sets
                    groupingSets = ast.groupBy.sets.map(set =>
                        set.map(col => typeof col === 'string' ? col : col.column)
                    );
                    // Collect all unique columns
                    const colSet = new Set();
                    for (const set of groupingSets) {
                        for (const col of set) colSet.add(col);
                    }
                    allColumns = [...colSet];
                }

                // Execute aggregation for each grouping set
                const allGroupedRows = [];
                for (const groupingSet of groupingSets) {
                    const groups = new Map();
                    for (const row of rows) {
                        const keyParts = groupingSet.map(col => String(row[col]));
                        const key = keyParts.join('|');
                        if (!groups.has(key)) {
                            groups.set(key, []);
                        }
                        groups.get(key).push(row);
                    }

                    // Process each group for this grouping set
                    for (const [, groupRows] of groups) {
                        const resultRow = {};
                        // Add all columns (null for rolled-up columns)
                        for (const col of allColumns) {
                            if (groupingSet.includes(col)) {
                                resultRow[col] = groupRows[0][col];
                                resultRow[`__grouping_${col}`] = 0; // Not rolled up
                            } else {
                                resultRow[col] = null;
                                resultRow[`__grouping_${col}`] = 1; // Rolled up (super-aggregate)
                            }
                        }
                        // Calculate aggregates
                        for (const col of ast.columns) {
                            if (col.type === 'aggregate') {
                                const rawName = `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                                const aggValue = calculateAggregate(col.func, col.arg, groupRows);
                                resultRow[rawName] = aggValue;
                                if (col.alias && col.alias !== rawName) {
                                    resultRow[col.alias] = aggValue;
                                }
                            }
                        }
                        allGroupedRows.push(resultRow);
                    }
                }
                rows = allGroupedRows;

                // Apply HAVING
                if (ast.having) {
                    rows = rows.filter(row => evalHaving(ast.having, row));
                }
            }
            // Handle aggregates without GROUP BY (whole table aggregate)
            else if (ast.columns.some(c => c.type === 'aggregate')) {
                const resultRow = {};
                for (const col of ast.columns) {
                    if (col.type === 'aggregate') {
                        const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                        resultRow[aggName] = calculateAggregate(col.func, col.arg, rows);
                    } else if (col.type === 'column') {
                        const colName = typeof col.value === 'string' ? col.value : col.value.column;
                        if (rows.length > 0) resultRow[colName] = rows[0][colName];
                    } else if (col.type === 'function') {
                        const alias = col.alias || `${col.func}(...)`;
                        if (rows.length > 0) resultRow[alias] = evaluateScalarFunction(col.func, col.args, rows[0], tableAliases);
                    } else if (col.type === 'arithmetic') {
                        const alias = col.alias || 'expr';
                        if (rows.length > 0) resultRow[alias] = evaluateArithmeticExpr(col.expr, rows[0], tableAliases);
                    } else if (col.type === 'literal') {
                        const alias = col.alias || 'value';
                        resultRow[alias] = col.value;
                    }
                }
                rows = [resultRow];
            }

            // Compute window functions (before ORDER BY but after all data is collected)
            const windowCols = ast.columns.filter(c => c.type === 'window');
            if (windowCols.length > 0) {
                rows = computeWindowFunctions(rows, windowCols, tableAliases);
            }

            // Apply QUALIFY (filter on window function results)
            if (ast.qualify) {
                rows = rows.filter(row => {
                    // QUALIFY can reference window function results by their alias
                    // The window function values are stored with __window_alias prefix
                    // Create a row view that maps aliases to window values
                    const rowWithWindowCols = { ...row };
                    for (const col of windowCols) {
                        const alias = col.alias || `${col.func}(...)`;
                        if (row[`__window_${alias}`] !== undefined) {
                            rowWithWindowCols[alias] = row[`__window_${alias}`];
                        }
                    }
                    return evalWhere(ast.qualify, rowWithWindowCols, tableAliases);
                });
            }

            // Apply ORDER BY first (before projection, using original column values)
            if (ast.orderBy && ast.orderBy.length > 0) {
                rows.sort((a, b) => {
                    for (const order of ast.orderBy) {
                        const col = typeof order.column === 'string' ? order.column : order.column.column;
                        const aVal = a[col];
                        const bVal = b[col];
                        const aNull = aVal == null;
                        const bNull = bVal == null;

                        // Handle NULL values
                        if (aNull || bNull) {
                            if (aNull && bNull) continue; // Both null, move to next column
                            // Determine nulls ordering: explicit setting or default (ASC=NULLS LAST, DESC=NULLS FIRST)
                            const nullsFirst = order.nullsFirst ?? order.desc;
                            if (aNull) return nullsFirst ? -1 : 1;
                            if (bNull) return nullsFirst ? 1 : -1;
                        }

                        if (aVal < bVal) return order.desc ? 1 : -1;
                        if (aVal > bVal) return order.desc ? -1 : 1;
                    }
                    return 0;
                });
            }

            // Apply OFFSET
            if (ast.offset) {
                rows = rows.slice(ast.offset);
            }

            // Apply LIMIT
            if (ast.limit) {
                rows = rows.slice(0, ast.limit);
            }

            // Project columns (after ORDER BY, OFFSET, LIMIT)
            let columnNames = [];
            const hasScalarSubquery = ast.columns.some(c => c.type === 'scalar_subquery');

            // Precompute aliases for each column (to handle duplicate default names)
            const columnAliases = [];
            let exprCount = 0, valueCount = 0;
            for (const col of ast.columns) {
                if (col.type === 'arithmetic' && !col.alias) {
                    exprCount++;
                    columnAliases.push(exprCount === 1 ? 'expr' : `expr${exprCount}`);
                } else if (col.type === 'literal' && !col.alias) {
                    valueCount++;
                    columnAliases.push(valueCount === 1 ? 'value' : `value${valueCount}`);
                } else {
                    columnAliases.push(null); // Use default logic
                }
            }

            if (!ast.columns.some(c => c.type === 'star') && !ast.groupBy) {
                // Use async projection if there are scalar subqueries
                const projectRow = async (row) => {
                    const result = {};
                    for (let colIdx = 0; colIdx < ast.columns.length; colIdx++) {
                        const col = ast.columns[colIdx];
                        if (col.type === 'column') {
                            const colName = typeof col.value === 'string' ? col.value : col.value.column;
                            const alias = col.alias || colName;
                            result[alias] = getColumnValue(row, col.value, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'aggregate') {
                            const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                            result[aggName] = row[aggName];
                            if (!columnNames.includes(aggName)) columnNames.push(aggName);
                        } else if (col.type === 'function') {
                            const alias = col.alias || `${col.func}(...)`;
                            result[alias] = evaluateScalarFunction(col.func, col.args, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'case') {
                            const alias = col.alias || 'case';
                            result[alias] = evaluateCaseExpr(col.expr, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'window') {
                            const alias = col.alias || `${col.func}(...)`;
                            result[alias] = row[`__window_${alias}`];
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'arithmetic') {
                            const alias = col.alias || columnAliases[colIdx] || 'expr';
                            result[alias] = evaluateArithmeticExpr(col.expr, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'literal') {
                            const alias = col.alias || columnAliases[colIdx] || 'value';
                            result[alias] = col.value;
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'scalar_subquery') {
                            // Correlated scalar subquery - execute for each row
                            const alias = col.alias || 'subquery';
                            result[alias] = await evaluateScalarSubquery(col.subquery, row, db, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        }
                    }
                    return result;
                };

                if (hasScalarSubquery) {
                    rows = await Promise.all(rows.map(projectRow));
                } else {
                    rows = rows.map(row => {
                        const result = {};
                        for (let colIdx = 0; colIdx < ast.columns.length; colIdx++) {
                            const col = ast.columns[colIdx];
                            if (col.type === 'column') {
                                const colName = typeof col.value === 'string' ? col.value : col.value.column;
                                const alias = col.alias || colName;
                                result[alias] = getColumnValue(row, col.value, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'aggregate') {
                                const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                                result[aggName] = row[aggName];
                                if (!columnNames.includes(aggName)) columnNames.push(aggName);
                            } else if (col.type === 'function') {
                                const alias = col.alias || `${col.func}(...)`;
                                result[alias] = evaluateScalarFunction(col.func, col.args, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'case') {
                                const alias = col.alias || 'case';
                                result[alias] = evaluateCaseExpr(col.expr, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'window') {
                                const alias = col.alias || `${col.func}(...)`;
                                result[alias] = row[`__window_${alias}`];
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'arithmetic') {
                                const alias = col.alias || columnAliases[colIdx] || 'expr';
                                result[alias] = evaluateArithmeticExpr(col.expr, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'literal') {
                                const alias = col.alias || columnAliases[colIdx] || 'value';
                                result[alias] = col.value;
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            }
                        }
                        return result;
                    });
                }
            } else if (rows.length > 0) {
                columnNames = Object.keys(rows[0]);
            }

            // Apply DISTINCT (after projection)
            if (ast.distinct) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: columnNames };
        }

        case 'UNION': {
            // Execute both sides recursively
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            let rows = [...leftResult.rows, ...rightResult.rows];

            // UNION (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'INTERSECT': {
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            const rightKeys = new Set(rightResult.rows.map(r => JSON.stringify(r)));
            let rows = leftResult.rows.filter(row => rightKeys.has(JSON.stringify(row)));

            // INTERSECT (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'EXCEPT': {
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            const rightKeys = new Set(rightResult.rows.map(r => JSON.stringify(r)));
            let rows = leftResult.rows.filter(row => !rightKeys.has(JSON.stringify(row)));

            // EXCEPT (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'WITH': {
            // Execute CTEs and create temporary tables
            const cteNames = [];

            // Helper to create temp CTE table
            const createCteTable = (name, rows, columns) => {
                const schema = columns.map(col => ({ name: col, type: 'TEXT' }));
                const rowsWithId = rows.map((row, i) => ({ ...row, __rowId: i }));
                db.tables.set(name, {
                    name,
                    schema,
                    fragments: [],
                    deletionVector: [],
                    rowCount: rowsWithId.length,
                    nextRowId: rowsWithId.length,
                    isCTE: true
                });
                db._writeBuffer.set(name, rowsWithId);
            };

            for (const cte of ast.ctes) {
                cteNames.push(cte.name);

                if (ast.recursive && cte.query.type === 'UNION' && cte.query.all) {
                    // Recursive CTE: UNION ALL structure
                    const anchor = cte.query.left;     // Base case
                    const recursive = cte.query.right; // Recursive part

                    // Execute anchor query (base case)
                    const anchorResult = await executeAST(db, anchor);
                    let currentRows = anchorResult.rows;
                    let allRows = [...currentRows];
                    const columns = anchorResult.columns;

                    // Iteratively execute recursive query until no new rows
                    const MAX_ITERATIONS = 10000;
                    let iteration = 0;

                    while (currentRows.length > 0 && allRows.length < MAX_ITERATIONS) {
                        iteration++;
                        // Create temp table with current iteration's rows
                        createCteTable(cte.name, currentRows, columns);

                        // Execute recursive query
                        const recursiveResult = await executeAST(db, recursive);
                        let newRows = recursiveResult.rows;

                        if (newRows.length === 0) break;

                        // Map recursive result columns to anchor column names
                        // This handles cases like "SELECT n + 1 FROM cte" where result is 'expr' not 'n'
                        const recursiveCols = recursiveResult.columns;
                        if (recursiveCols.length === columns.length) {
                            newRows = newRows.map(row => {
                                const mappedRow = {};
                                for (let i = 0; i < columns.length; i++) {
                                    const anchorCol = columns[i];
                                    const recCol = recursiveCols[i];
                                    mappedRow[anchorCol] = row[recCol];
                                }
                                return mappedRow;
                            });
                        }

                        allRows.push(...newRows);
                        currentRows = newRows;
                    }

                    // Create final CTE table with all accumulated rows
                    createCteTable(cte.name, allRows, columns);
                } else {
                    // Non-recursive CTE
                    const cteResult = await executeAST(db, cte.query);
                    createCteTable(cte.name, cteResult.rows, cteResult.columns);
                }
            }

            try {
                // Execute main query
                const result = await executeAST(db, ast.query);
                return result;
            } finally {
                // Clean up CTE tables
                for (const name of cteNames) {
                    db.tables.delete(name);
                    db._writeBuffer.delete(name);
                }
            }
        }

        case 'EXPLAIN': {
            const plan = generateQueryPlan(ast.statement);

            if (!ast.analyze) {
                // EXPLAIN without ANALYZE - just return the plan
                return { columns: ['plan'], rows: [[JSON.stringify(plan)]] };
            }

            // EXPLAIN ANALYZE - execute and measure timing
            const startTime = performance.now();
            const result = await executeAST(db, ast.statement);
            const elapsed = performance.now() - startTime;

            const execution = {
                actualTimeMs: Math.round(elapsed * 1000) / 1000, // Round to 3 decimal places
                rowsReturned: result.rows ? result.rows.length : 0,
                rowsTotal: result.rows ? result.rows.length : 0
            };

            return {
                columns: ['plan_with_execution'],
                rows: [[JSON.stringify({ plan, execution })]]
            };
        }

        case 'PIVOT': {
            // Execute the SELECT first
            const selectResult = await executeAST(db, ast.select);
            const rows = selectResult.rows;

            // Determine grouping columns (all SELECT columns except pivot and value columns)
            const selectCols = ast.select.columns
                .filter(c => c.type === 'column')
                .map(c => typeof c.value === 'string' ? c.value : c.value.column);
            const groupCols = selectCols.filter(
                c => c !== ast.pivotColumn && c !== ast.valueColumn
            );

            // Group rows by grouping columns
            const groups = new Map();
            for (const row of rows) {
                const keyParts = groupCols.map(c => String(row[c]));
                const key = keyParts.join('|');
                if (!groups.has(key)) {
                    groups.set(key, { keyRow: row, pivotData: {} });
                }
                const pivotVal = row[ast.pivotColumn];
                const dataVal = row[ast.valueColumn];
                if (!groups.get(key).pivotData[pivotVal]) {
                    groups.get(key).pivotData[pivotVal] = [];
                }
                groups.get(key).pivotData[pivotVal].push(dataVal);
            }

            // Aggregate function helper
            const aggregate = (func, values) => {
                if (!values || values.length === 0) return 0;
                const nums = values.filter(v => v != null).map(v => Number(v));
                switch (func) {
                    case 'SUM': return nums.reduce((a, b) => a + b, 0);
                    case 'COUNT': return nums.length;
                    case 'AVG': return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
                    case 'MIN': return Math.min(...nums);
                    case 'MAX': return Math.max(...nums);
                    default: return nums[0];
                }
            };

            // Build result rows
            const resultRows = [];
            for (const [, { keyRow, pivotData }] of groups) {
                const newRow = {};
                for (const col of groupCols) {
                    newRow[col] = keyRow[col];
                }
                for (const pv of ast.pivotValues) {
                    newRow[pv] = aggregate(ast.aggFunc, pivotData[pv] || []);
                }
                resultRows.push(newRow);
            }

            return { columns: [...groupCols, ...ast.pivotValues], rows: resultRows };
        }

        case 'UNPIVOT': {
            // Execute the SELECT first
            const selectResult = await executeAST(db, ast.select);
            const rows = selectResult.rows;

            // Determine preserved columns (all SELECT columns except unpivot columns)
            const selectCols = ast.select.columns
                .filter(c => c.type === 'column')
                .map(c => typeof c.value === 'string' ? c.value : c.value.column);
            const preservedCols = selectCols.filter(c => !ast.unpivotColumns.includes(c));

            // Build result rows
            const resultRows = [];
            for (const row of rows) {
                for (const col of ast.unpivotColumns) {
                    const val = row[col];
                    if (val != null) { // Skip NULL values
                        const newRow = {};
                        for (const pc of preservedCols) {
                            newRow[pc] = row[pc];
                        }
                        newRow[ast.nameColumn] = col;
                        newRow[ast.valueColumn] = val;
                        resultRows.push(newRow);
                    }
                }
            }

            return { columns: [...preservedCols, ast.nameColumn, ast.valueColumn], rows: resultRows };
        }

        default:
            throw new Error(`Unknown statement type: ${ast.type}`);
    }
}

// ============================================================================
// WorkerVault - Unified vault storage (KV + tables)
// ============================================================================


export { executeSQL, executeAST, evalWhere, getColumnValue, evaluateArithmeticExpr, calculateAggregate };
