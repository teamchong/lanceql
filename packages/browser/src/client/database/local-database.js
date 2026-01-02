/**
 * LocalDatabase - OPFS-backed local database with join optimization
 */

class OPFSJoinExecutor {
    constructor(storage = opfsStorage) {
        this.storage = storage;
        this.sessionId = `join_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        this.basePath = `_join_temp/${this.sessionId}`;
        this.numPartitions = 64;  // Number of hash partitions
        this.chunkSize = 1000;    // Rows per chunk when streaming
        this.stats = {
            leftRowsWritten: 0,
            rightRowsWritten: 0,
            resultRowsWritten: 0,
            bytesWrittenToOPFS: 0,
            bytesReadFromOPFS: 0,
            partitionsUsed: new Set(),
        };
    }

    /**
     * Execute a hash join using OPFS for intermediate storage
     * @param {AsyncGenerator} leftStream - Async generator yielding {columns, rows} chunks
     * @param {AsyncGenerator} rightStream - Async generator yielding {columns, rows} chunks
     * @param {string} leftKey - Join key column name for left table
     * @param {string} rightKey - Join key column name for right table
     * @param {Object} options - Join options
     * @returns {AsyncGenerator} Yields result chunks
     */
    async *executeHashJoin(leftStream, rightStream, leftKey, rightKey, options = {}) {
        const {
            limit = Infinity,
            projection = null,
            leftAlias = 'left',
            rightAlias = 'right',
            joinType = 'INNER',
            prePartitionedLeft = null  // Optional: pre-partitioned left metadata for semi-join optimization
        } = options;

        try {
            let leftMeta;
            if (prePartitionedLeft) {
                leftMeta = prePartitionedLeft;
            } else {
                leftMeta = await this._partitionToOPFS(leftStream, leftKey, 'left');
            }

            const rightMeta = await this._partitionToOPFS(rightStream, rightKey, 'right');

            let totalYielded = 0;

            // Create NULL padding arrays for outer joins
            const leftNulls = new Array(leftMeta.columns.length).fill(null);
            const rightNulls = new Array(rightMeta.columns.length).fill(null);

            // Helper to build result columns
            const resultColumns = [
                ...leftMeta.columns.map(c => `${leftAlias}.${c}`),
                ...rightMeta.columns.map(c => `${rightAlias}.${c}`)
            ];

            // Helper to yield a chunk
            const yieldChunk = function*(chunk) {
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk.splice(0) };
                }
            };

            if (joinType === 'CROSS') {
                const chunk = [];

                // Load all partitions for both tables
                for (const leftPartitionId of leftMeta.partitionsUsed) {
                    const leftPartition = await this._loadPartition('left', leftPartitionId, leftMeta.columns);
                    for (const rightPartitionId of rightMeta.partitionsUsed) {
                        const rightPartition = await this._loadPartition('right', rightPartitionId, rightMeta.columns);
                        for (const leftRow of leftPartition) {
                            for (const rightRow of rightPartition) {
                                if (totalYielded >= limit) break;
                                chunk.push([...leftRow, ...rightRow]);
                                totalYielded++;
                                if (chunk.length >= this.chunkSize) {
                                    yield { columns: resultColumns, rows: chunk.splice(0) };
                                }
                            }
                            if (totalYielded >= limit) break;
                        }
                        if (totalYielded >= limit) break;
                    }
                    if (totalYielded >= limit) break;
                }
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk };
                }
                return;
            }

            // For hash-based joins, determine which partitions to process
            const leftKeyIndex = leftMeta.columns.indexOf(leftKey);
            const rightKeyIndex = rightMeta.columns.indexOf(rightKey);

            // Track matched rows for outer joins
            const isLeftOuter = joinType === 'LEFT' || joinType === 'FULL';
            const isRightOuter = joinType === 'RIGHT' || joinType === 'FULL';

            // For RIGHT/FULL: track which right rows have been matched (by partition_rowIndex)
            const matchedRightRows = isRightOuter ? new Set() : null;

            // Process partitions that have data on both sides (for INNER and outer join matches)
            const bothSidesPartitions = new Set(
                [...leftMeta.partitionsUsed].filter(p => rightMeta.partitionsUsed.has(p))
            );

            for (const partitionId of bothSidesPartitions) {
                if (totalYielded >= limit) break;

                // Load left partition into memory
                const leftPartition = await this._loadPartition('left', partitionId, leftMeta.columns);
                if (leftPartition.length === 0) continue;

                // Build hash map and track matched left rows for LEFT/FULL JOIN
                const hashMap = new Map();
                const matchedLeftIndices = isLeftOuter ? new Set() : null;

                for (let i = 0; i < leftPartition.length; i++) {
                    const row = leftPartition[i];
                    const key = row[leftKeyIndex];
                    if (key !== null && key !== undefined) {
                        if (!hashMap.has(key)) hashMap.set(key, []);
                        hashMap.get(key).push({ row, index: i });
                    }
                }

                // Load right partition and probe hash map
                const rightPartition = await this._loadPartition('right', partitionId, rightMeta.columns);

                const chunk = [];
                for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
                    if (totalYielded >= limit) break;

                    const rightRow = rightPartition[rightIdx];
                    const key = rightRow[rightKeyIndex];
                    const leftEntries = hashMap.get(key);

                    if (leftEntries) {
                        // Track that this right row matched (for RIGHT/FULL)
                        if (matchedRightRows) {
                            matchedRightRows.add(`${partitionId}_${rightIdx}`);
                        }

                        for (const { row: leftRow, index: leftIdx } of leftEntries) {
                            if (totalYielded >= limit) break;

                            // Track that this left row matched (for LEFT/FULL)
                            if (matchedLeftIndices) {
                                matchedLeftIndices.add(leftIdx);
                            }

                            chunk.push([...leftRow, ...rightRow]);
                            totalYielded++;

                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }
                }

                // For LEFT/FULL: emit unmatched left rows with NULL right side
                if (isLeftOuter && matchedLeftIndices) {
                    for (let i = 0; i < leftPartition.length; i++) {
                        if (totalYielded >= limit) break;
                        if (!matchedLeftIndices.has(i)) {
                            chunk.push([...leftPartition[i], ...rightNulls]);
                            totalYielded++;
                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }
                }

                // Yield remaining rows in chunk
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk };
                }
            }

            // For LEFT/FULL: handle left-only partitions (no matching right data)
            if (isLeftOuter) {
                for (const partitionId of leftMeta.partitionsUsed) {
                    if (totalYielded >= limit) break;
                    if (bothSidesPartitions.has(partitionId)) continue;  // Already processed

                    const leftPartition = await this._loadPartition('left', partitionId, leftMeta.columns);
                    const chunk = [];
                    for (const leftRow of leftPartition) {
                        if (totalYielded >= limit) break;
                        chunk.push([...leftRow, ...rightNulls]);
                        totalYielded++;
                        if (chunk.length >= this.chunkSize) {
                            yield { columns: resultColumns, rows: chunk.splice(0) };
                        }
                    }
                    if (chunk.length > 0) {
                        yield { columns: resultColumns, rows: chunk };
                    }
                }
            }

            // For RIGHT/FULL: emit unmatched right rows with NULL left side
            if (isRightOuter) {
                for (const partitionId of rightMeta.partitionsUsed) {
                    if (totalYielded >= limit) break;

                    const rightPartition = await this._loadPartition('right', partitionId, rightMeta.columns);
                    const chunk = [];

                    for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
                        if (totalYielded >= limit) break;

                        // Check if this row was matched during the main join
                        const rowKey = `${partitionId}_${rightIdx}`;
                        if (!matchedRightRows.has(rowKey)) {
                            chunk.push([...leftNulls, ...rightPartition[rightIdx]]);
                            totalYielded++;
                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }

                    if (chunk.length > 0) {
                        yield { columns: resultColumns, rows: chunk };
                    }
                }
            }

        } finally {
            // Cleanup temp files
            await this.cleanup();
        }
    }

    /**
     * Partition a stream of data to OPFS files by hash(key)
     * @param {AsyncGenerator} stream - Data stream to partition
     * @param {string} keyColumn - Column name to use as partition key
     * @param {string} side - 'left' or 'right' table
     * @param {boolean} collectKeys - If true, collect unique key values for semi-join optimization
     */
    async _partitionToOPFS(stream, keyColumn, side, collectKeys = false) {
        const partitionBuffers = new Map();  // partitionId -> rows[]
        const flushThreshold = 500;  // Flush to OPFS when buffer reaches this size
        let columns = null;
        let keyIndex = -1;
        let totalRows = 0;
        const partitionsUsed = new Set();
        const collectedKeys = collectKeys ? new Set() : null;

        for await (const chunk of stream) {
            if (!columns) {
                columns = chunk.columns;
                keyIndex = columns.indexOf(keyColumn);
                if (keyIndex === -1) {
                    throw new Error(`Join key column '${keyColumn}' not found in columns: ${columns.join(', ')}`);
                }
            }

            for (const row of chunk.rows) {
                const key = row[keyIndex];
                const partitionId = this._hashToPartition(key);
                partitionsUsed.add(partitionId);

                // Collect unique keys for semi-join optimization
                if (collectKeys && key !== null && key !== undefined) {
                    collectedKeys.add(key);
                }

                if (!partitionBuffers.has(partitionId)) {
                    partitionBuffers.set(partitionId, []);
                }
                partitionBuffers.get(partitionId).push(row);
                totalRows++;

                // Flush partition buffer if too large
                if (partitionBuffers.get(partitionId).length >= flushThreshold) {
                    await this._appendToPartition(side, partitionId, partitionBuffers.get(partitionId));
                    partitionBuffers.set(partitionId, []);
                }
            }
        }

        // Flush remaining buffers
        for (const [partitionId, rows] of partitionBuffers) {
            if (rows.length > 0) {
                await this._appendToPartition(side, partitionId, rows);
            }
        }

        if (side === 'left') {
            this.stats.leftRowsWritten = totalRows;
        } else {
            this.stats.rightRowsWritten = totalRows;
        }

        return { columns, totalRows, partitionsUsed, collectedKeys };
    }

    /**
     * Hash a value to a partition number
     */
    _hashToPartition(value) {
        if (value === null || value === undefined) {
            return 0;  // Null keys go to partition 0
        }

        // Simple string hash
        const str = String(value);
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;  // Convert to 32-bit integer
        }
        return Math.abs(hash) % this.numPartitions;
    }

    /**
     * Append rows to a partition file in OPFS
     */
    async _appendToPartition(side, partitionId, rows) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, '0')}.jsonl`;

        // Serialize rows as JSONL (one JSON array per line)
        const jsonl = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
        const data = new TextEncoder().encode(jsonl);

        // Load existing data if any
        const existing = await this.storage.load(path);

        if (existing) {
            // Append to existing
            const combined = new Uint8Array(existing.length + data.length);
            combined.set(existing);
            combined.set(data, existing.length);
            await this.storage.save(path, combined);
            this.stats.bytesWrittenToOPFS += data.length;
        } else {
            await this.storage.save(path, data);
            this.stats.bytesWrittenToOPFS += data.length;
        }

        this.stats.partitionsUsed.add(partitionId);
    }

    /**
     * Load a partition from OPFS
     */
    async _loadPartition(side, partitionId, columns) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, '0')}.jsonl`;

        const data = await this.storage.load(path);
        if (!data) return [];

        this.stats.bytesReadFromOPFS += data.length;

        const text = new TextDecoder().decode(data);
        const lines = text.trim().split('\n').filter(line => line);

        return lines.map(line => JSON.parse(line));
    }

    /**
     * Get execution statistics
     */
    getStats() {
        return {
            ...this.stats,
            partitionsUsed: this.stats.partitionsUsed.size,
            bytesWrittenMB: (this.stats.bytesWrittenToOPFS / 1024 / 1024).toFixed(2),
            bytesReadMB: (this.stats.bytesReadFromOPFS / 1024 / 1024).toFixed(2),
        };
    }

    /**
     * Cleanup temp files
     */
    async cleanup() {
        try {
            await this.storage.deleteDir(this.basePath);
        } catch {
            // Cleanup failures are non-fatal
        }
    }
}

// =============================================================================
// LocalDatabase - ACID-compliant local database with CRUD support
// =============================================================================

/**
 * Data types for schema definition
 */
const DataType = {
    INT: 'int64',
    INTEGER: 'int64',
    BIGINT: 'int64',
    FLOAT: 'float32',
    REAL: 'float64',
    DOUBLE: 'float64',
    TEXT: 'string',
    VARCHAR: 'string',
    BOOLEAN: 'bool',
    BOOL: 'bool',
    VECTOR: 'vector',
};

/**
 * LocalDatabase - ACID-compliant database stored in IndexedDB/OPFS
 *
 * Uses manifest-based versioning for ACID:
 * - Atomicity: Manifest update is atomic
 * - Consistency: Always read from valid manifest
 * - Isolation: Each transaction sees snapshot
 * - Durability: Persisted to OPFS (Origin Private File System)
 *
 * Storage: Uses OPFS exclusively for all data sizes. No IndexedDB migration needed.
 *
 * Write Buffer: Inserts are buffered in memory and flushed to OPFS periodically
 * for high-throughput writes without exhausting file handles.
 */
class LocalDatabase {
    /**
     * Create a LocalDatabase instance.
     * All operations are executed in a SharedWorker for OPFS sync access.
     *
     * @param {string} name - Database name
     */
    constructor(name) {
        this.name = name;
        this._ready = false;
    }

    /**
     * Open or create the database.
     * Connects to the SharedWorker.
     *
     * @returns {Promise<LocalDatabase>}
     */
    async open() {
        if (this._ready) return this;
        await workerRPC('db:open', { name: this.name });
        this._ready = true;
        return this;
    }

    async _ensureOpen() {
        if (!this._ready) {
            await this.open();
        }
    }

    /**
     * CREATE TABLE
     * @param {string} tableName - Table name
     * @param {Array} columns - Column definitions [{name, type, primaryKey?, vectorDim?}]
     * @param {boolean} ifNotExists - If true, don't error if table already exists
     * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
     */
    async createTable(tableName, columns, ifNotExists = false) {
        await this._ensureOpen();
        return workerRPC('db:createTable', {
            db: this.name,
            tableName,
            columns,
            ifNotExists
        });
    }

    /**
     * DROP TABLE
     * @param {string} tableName - Table name
     * @param {boolean} ifExists - If true, don't error if table doesn't exist
     * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
     */
    async dropTable(tableName, ifExists = false) {
        await this._ensureOpen();
        return workerRPC('db:dropTable', {
            db: this.name,
            tableName,
            ifExists
        });
    }

    /**
     * INSERT INTO
     * @param {string} tableName - Table name
     * @param {Array} rows - Array of row objects [{col1: val1, col2: val2}, ...]
     * @returns {Promise<{success: boolean, inserted: number}>}
     */
    async insert(tableName, rows) {
        await this._ensureOpen();
        return workerRPC('db:insert', {
            db: this.name,
            tableName,
            rows
        });
    }

    /**
     * Flush all buffered writes to OPFS
     * @returns {Promise<void>}
     */
    async flush() {
        await this._ensureOpen();
        return workerRPC('db:flush', { db: this.name });
    }

    /**
     * DELETE FROM
     * @param {string} tableName - Table name
     * @param {Object} where - WHERE clause as parsed AST (column/op/value)
     * @returns {Promise<{success: boolean, deleted: number}>}
     */
    async delete(tableName, where = null) {
        await this._ensureOpen();
        return workerRPC('db:delete', {
            db: this.name,
            tableName,
            where
        });
    }

    /**
     * UPDATE
     * @param {string} tableName - Table name
     * @param {Object} updates - Column updates {col1: newVal, col2: newVal}
     * @param {Object} where - WHERE clause as parsed AST (column/op/value)
     * @returns {Promise<{success: boolean, updated: number}>}
     */
    async update(tableName, updates, where = null) {
        await this._ensureOpen();
        return workerRPC('db:update', {
            db: this.name,
            tableName,
            updates,
            where
        });
    }

    /**
     * SELECT (query)
     * @param {string} tableName - Table name
     * @param {Object} options - Query options {columns, where, limit, offset, orderBy}
     * @returns {Promise<Array>}
     */
    async select(tableName, options = {}) {
        await this._ensureOpen();

        // Convert where function to AST if needed (cannot serialize functions)
        const rpcOptions = { ...options };
        delete rpcOptions.where; // Functions can't be serialized

        return workerRPC('db:select', {
            db: this.name,
            tableName,
            options: rpcOptions,
            where: options.whereAST || null
        });
    }

    /**
     * Execute SQL statement
     * @param {string} sql - SQL statement
     * @returns {Promise<any>} Result
     */
    async exec(sql) {
        await this._ensureOpen();
        return workerRPC('db:exec', { db: this.name, sql });
    }

    /**
     * Get table info
     * @param {string} tableName - Table name
     * @returns {Promise<Object>} Table state
     */
    async getTable(tableName) {
        await this._ensureOpen();
        return workerRPC('db:getTable', { db: this.name, tableName });
    }

    /**
     * List all tables
     * @returns {Promise<string[]>} Table names
     */
    async listTables() {
        await this._ensureOpen();
        return workerRPC('db:listTables', { db: this.name });
    }

    /**
     * Compact the database (merge fragments, remove deleted rows)
     * @returns {Promise<{success: boolean, compacted: number}>}
     */
    async compact() {
        await this._ensureOpen();
        return workerRPC('db:compact', { db: this.name });
    }

    /**
     * Streaming scan - yields batches of rows for memory-efficient processing
     * @param {string} tableName - Table name
     * @param {Object} options - Scan options {batchSize, columns}
     * @yields {Object[]} Batch of rows
     *
     * @example
     * for await (const batch of db.scan('users', { batchSize: 1000 })) {
     *   processBatch(batch);
     * }
     */
    async *scan(tableName, options = {}) {
        await this._ensureOpen();

        // Start scan stream
        const streamId = await workerRPC('db:scanStart', {
            db: this.name,
            tableName,
            options
        });

        // Iterate through batches
        while (true) {
            const { batch, done } = await workerRPC('db:scanNext', {
                db: this.name,
                streamId
            });

            if (batch.length > 0) {
                yield batch;
            }

            if (done) break;
        }
    }

    /**
     * Close the database
     * @returns {Promise<void>}
     */
    async close() {
        await this._ensureOpen();
        await this.flush();
    }
}


export { OPFSJoinExecutor, LocalDatabase };
