/**
 * OPFS Result Buffer
 *
 * Streams large operation results to OPFS to avoid OOM.
 * Results can be read back as an async iterator without loading all into memory.
 *
 * @example
 * const buffer = new OPFSResultBuffer('join-results');
 * await buffer.init();
 *
 * // Write chunks
 * await buffer.appendMatches(new Uint32Array([0, 1, 2, 3]));
 * await buffer.appendMatches(new Uint32Array([4, 5, 6, 7]));
 * await buffer.finalize();
 *
 * // Stream results
 * for await (const chunk of buffer.stream()) {
 *     console.log(chunk); // Uint32Array chunks
 * }
 */

const OPFS_DIR = '.lanceql-buffers';
const CHUNK_SIZE = 64 * 1024; // 64KB read chunks

export class OPFSResultBuffer {
    /**
     * @param {string} name - Unique name for this buffer
     * @param {Object} options
     * @param {number} options.maxMemory - Max memory before flush (default 16MB)
     */
    constructor(name, options = {}) {
        this.name = name;
        this.maxMemory = options.maxMemory || 16 * 1024 * 1024;

        this._file = null;
        this._syncHandle = null;
        this._writeOffset = 0;
        this._memBuffer = [];
        this._memSize = 0;
        this._finalized = false;
        this._totalEntries = 0;
        this._entrySize = 4; // Default: u32
    }

    /**
     * Initialize OPFS file handle.
     * @param {number} entrySize - Bytes per entry (4 for u32, 8 for u64)
     */
    async init(entrySize = 4) {
        this._entrySize = entrySize;

        try {
            const root = await navigator.storage.getDirectory();
            const dir = await root.getDirectoryHandle(OPFS_DIR, { create: true });

            // Remove existing file if any
            try {
                await dir.removeEntry(this.name);
            } catch (e) {
                // File doesn't exist, that's fine
            }

            this._file = await dir.getFileHandle(this.name, { create: true });
            this._syncHandle = await this._file.createSyncAccessHandle();
            this._writeOffset = 0;
            this._finalized = false;
            this._totalEntries = 0;
        } catch (e) {
            console.warn('[OPFSResultBuffer] OPFS not available, using memory-only mode');
            this._syncHandle = null;
        }
    }

    /**
     * Append match pairs (or any typed array data).
     * @param {Uint32Array|BigUint64Array} data - Data to append
     */
    async appendMatches(data) {
        if (this._finalized) {
            throw new Error('Buffer already finalized');
        }

        this._totalEntries += data.length;

        // Memory-only mode (OPFS unavailable)
        if (!this._syncHandle) {
            this._memBuffer.push(data.slice());
            return;
        }

        // Add to memory buffer
        this._memBuffer.push(data);
        this._memSize += data.byteLength;

        // Flush to OPFS when memory threshold exceeded
        if (this._memSize >= this.maxMemory) {
            await this._flush();
        }
    }

    /**
     * Flush memory buffer to OPFS.
     */
    async _flush() {
        if (!this._syncHandle || this._memBuffer.length === 0) return;

        // Concatenate all buffers
        const totalSize = this._memBuffer.reduce((sum, b) => sum + b.byteLength, 0);
        const combined = new Uint8Array(totalSize);
        let offset = 0;
        for (const buf of this._memBuffer) {
            combined.set(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), offset);
            offset += buf.byteLength;
        }

        // Write to OPFS
        this._syncHandle.write(combined, { at: this._writeOffset });
        this._writeOffset += totalSize;

        // Clear memory
        this._memBuffer = [];
        this._memSize = 0;
    }

    /**
     * Finalize the buffer (flush remaining data).
     * @returns {Object} Stats about the buffer
     */
    async finalize() {
        if (this._finalized) return this.stats();

        await this._flush();
        this._finalized = true;

        return this.stats();
    }

    /**
     * Get buffer statistics.
     */
    stats() {
        return {
            totalEntries: this._totalEntries,
            totalBytes: this._writeOffset + this._memSize,
            onDisk: this._writeOffset,
            inMemory: this._memSize,
            entrySize: this._entrySize
        };
    }

    /**
     * Stream results as async iterator.
     * @param {number} chunkSize - Entries per chunk (default: 16K entries)
     * @yields {Uint32Array|BigUint64Array} Chunks of data
     */
    async *stream(chunkSize = 16384) {
        const bytesPerChunk = chunkSize * this._entrySize;
        const ArrayType = this._entrySize === 8 ? BigUint64Array : Uint32Array;

        // Read from OPFS
        if (this._syncHandle && this._writeOffset > 0) {
            let readOffset = 0;
            while (readOffset < this._writeOffset) {
                const remaining = this._writeOffset - readOffset;
                const toRead = Math.min(bytesPerChunk, remaining);
                const buffer = new ArrayBuffer(toRead);
                const view = new Uint8Array(buffer);

                this._syncHandle.read(view, { at: readOffset });
                readOffset += toRead;

                yield new ArrayType(buffer);
            }
        }

        // Yield remaining memory buffers
        for (const buf of this._memBuffer) {
            yield buf;
        }
    }

    /**
     * Read all results into a single array.
     * WARNING: May OOM for large results. Prefer stream() for large data.
     * @returns {Uint32Array|BigUint64Array}
     */
    async readAll() {
        const ArrayType = this._entrySize === 8 ? BigUint64Array : Uint32Array;
        const totalBytes = this._writeOffset + this._memBuffer.reduce((s, b) => s + b.byteLength, 0);
        const result = new ArrayType(totalBytes / this._entrySize);

        let writeIdx = 0;
        for await (const chunk of this.stream()) {
            result.set(chunk, writeIdx);
            writeIdx += chunk.length;
        }

        return result;
    }

    /**
     * Close and cleanup.
     * @param {boolean} deleteFile - Whether to delete the OPFS file
     */
    async close(deleteFile = true) {
        if (this._syncHandle) {
            this._syncHandle.close();
            this._syncHandle = null;
        }

        if (deleteFile && this._file) {
            try {
                const root = await navigator.storage.getDirectory();
                const dir = await root.getDirectoryHandle(OPFS_DIR);
                await dir.removeEntry(this.name);
            } catch (e) {
                // Ignore cleanup errors
            }
        }

        this._memBuffer = [];
        this._memSize = 0;
    }
}

/**
 * Create a temporary result buffer that auto-cleans.
 * @param {string} prefix - Name prefix
 * @returns {OPFSResultBuffer}
 */
export function createTempBuffer(prefix = 'temp') {
    const name = `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    return new OPFSResultBuffer(name);
}
