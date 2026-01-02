/**
 * Data Types and Lance File Writer
 */

// Text encoder/decoder
export const E = new TextEncoder();
export const D = new TextDecoder();

export const DataType = {
    INT64: 'int64',
    INT32: 'int32',
    FLOAT64: 'float64',
    FLOAT32: 'float32',
    STRING: 'string',
    BOOL: 'bool',
    VECTOR: 'vector',
};

export class LanceFileWriter {
    constructor(schema) {
        this.schema = schema;
        this.columns = new Map();
        this.rowCount = 0;
    }

    addRows(rows) {
        for (const row of rows) {
            for (const col of this.schema) {
                if (!this.columns.has(col.name)) {
                    this.columns.set(col.name, []);
                }
                this.columns.get(col.name).push(row[col.name] ?? null);
            }
            this.rowCount++;
        }
    }

    build() {
        // JSON columnar format (reliable fallback)
        const data = {
            format: 'json',
            schema: this.schema,
            columns: {},
            rowCount: this.rowCount,
        };

        for (const [name, values] of this.columns) {
            data.columns[name] = values;
        }

        return E.encode(JSON.stringify(data));
    }
}
