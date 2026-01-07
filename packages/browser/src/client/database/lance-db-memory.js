/**
 * LanceDatabase - Memory table operations
 * Extracted from lance-database.js for modularity
 */

import { SQLExecutor } from '../sql/executor.js';

/**
 * In-memory table for CRUD operations
 */
export class MemoryTable {
    constructor(name, schema) {
        this.name = name;
        this.schema = schema;
        this.columns = schema.map(c => c.name);
        this.rows = [];
        this._columnIndex = new Map();
        this.columns.forEach((col, idx) => {
            this._columnIndex.set(col.toLowerCase(), idx);
        });
    }

    /**
     * Convert to in-memory data format for executor
     */
    toInMemoryData() {
        // Return format expected by SQLExecutor._executeOnInMemoryData
        return { columns: this.columns, rows: this.rows };
    }
}

/**
 * Execute CREATE TABLE - creates an in-memory table
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed CREATE TABLE AST
 * @returns {Object} Result with success flag
 */
export function executeCreateTable(db, ast) {
    const tableName = (ast.table || ast.name || '').toLowerCase();

    if (!tableName) {
        throw new Error('CREATE TABLE requires a table name');
    }

    if (db.memoryTables.has(tableName) || db.tables.has(tableName)) {
        if (ast.ifNotExists) {
            return { success: true, existed: true, table: tableName };
        }
        throw new Error(`Table '${tableName}' already exists`);
    }

    const schema = (ast.columns || []).map(col => ({
        name: col.name,
        dataType: col.dataType || col.type || 'TEXT',
        primaryKey: col.primaryKey || false
    }));

    if (schema.length === 0) {
        throw new Error('CREATE TABLE requires at least one column');
    }

    const table = new MemoryTable(tableName, schema);
    db.memoryTables.set(tableName, table);

    return {
        success: true,
        table: tableName,
        columns: schema.map(c => c.name)
    };
}

/**
 * Execute DROP TABLE - removes an in-memory table
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed DROP TABLE AST
 * @returns {Object} Result with success flag
 */
export function executeDropTable(db, ast) {
    const tableName = (ast.table || ast.name || '').toLowerCase();

    if (!db.memoryTables.has(tableName)) {
        if (ast.ifExists) {
            return { success: true, existed: false, table: tableName };
        }
        throw new Error(`Memory table '${tableName}' not found`);
    }

    db.memoryTables.delete(tableName);
    return { success: true, table: tableName };
}

/**
 * Execute INSERT - adds rows to a memory table
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed INSERT AST
 * @returns {Object} Result with inserted count
 */
export function executeInsert(db, ast) {
    const tableName = (ast.table || '').toLowerCase();
    const table = db.memoryTables.get(tableName);

    if (!table) {
        throw new Error(`Memory table '${tableName}' not found. Use CREATE TABLE first.`);
    }

    const insertCols = ast.columns || table.columns;
    let inserted = 0;

    for (const astRow of (ast.rows || ast.values || [])) {
        const row = new Array(table.columns.length).fill(null);

        insertCols.forEach((colName, i) => {
            const colIdx = table._columnIndex.get(
                (typeof colName === 'string' ? colName : colName.name || colName).toLowerCase()
            );
            if (colIdx !== undefined && i < astRow.length) {
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
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed UPDATE AST
 * @returns {Object} Result with updated count
 */
export function executeUpdate(db, ast) {
    const tableName = (ast.table || '').toLowerCase();
    const table = db.memoryTables.get(tableName);

    if (!table) {
        throw new Error(`Memory table '${tableName}' not found`);
    }

    const columnData = {};
    table.columns.forEach((col, idx) => {
        columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
    });

    const executor = new SQLExecutor({ columnNames: table.columns });
    let updated = 0;

    for (let i = 0; i < table.rows.length; i++) {
        const matches = !ast.where || executor._evaluateInMemoryExpr(ast.where, columnData, i);

        if (matches) {
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
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed DELETE AST
 * @returns {Object} Result with deleted count
 */
export function executeDelete(db, ast) {
    const tableName = (ast.table || '').toLowerCase();
    const table = db.memoryTables.get(tableName);

    if (!table) {
        throw new Error(`Memory table '${tableName}' not found`);
    }

    const originalCount = table.rows.length;

    if (ast.where) {
        const columnData = {};
        table.columns.forEach((col, idx) => {
            columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
        });

        const executor = new SQLExecutor({ columnNames: table.columns });

        table.rows = table.rows.filter((_, i) =>
            !executor._evaluateInMemoryExpr(ast.where, columnData, i)
        );
    } else {
        table.rows = [];
    }

    return {
        success: true,
        deleted: originalCount - table.rows.length,
        remaining: table.rows.length
    };
}
