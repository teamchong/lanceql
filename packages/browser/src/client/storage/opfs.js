/**
 * OPFS Storage - File system operations using Origin Private File System
 */

class OPFSStorage {
    constructor(rootDir = 'lanceql') {
        this.rootDir = rootDir;
        this.root = null;
    }

    /**
     * Get OPFS root directory, creating if needed
     */
    async getRoot() {
        if (this.root) return this.root;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            throw new Error('OPFS not available. Requires modern browser with Origin Private File System support.');
        }

        const opfsRoot = await navigator.storage.getDirectory();
        this.root = await opfsRoot.getDirectoryHandle(this.rootDir, { create: true });
        return this.root;
    }

    async open() {
        await this.getRoot();
        return this;
    }

    /**
     * Get or create a subdirectory
     */
    async getDir(path) {
        const root = await this.getRoot();
        const parts = path.split('/').filter(p => p);

        let current = root;
        for (const part of parts) {
            current = await current.getDirectoryHandle(part, { create: true });
        }
        return current;
    }

    /**
     * Save data to a file
     * @param {string} path - File path (e.g., 'mydb/users/frag_001.lance')
     * @param {Uint8Array} data - File data
     */
    async save(path, data) {
        const parts = path.split('/');
        const fileName = parts.pop();
        const dirPath = parts.join('/');

        const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
        const fileHandle = await dir.getFileHandle(fileName, { create: true });

        // Use sync access handle for better performance if available
        if (fileHandle.createSyncAccessHandle) {
            try {
                const accessHandle = await fileHandle.createSyncAccessHandle();
                accessHandle.truncate(0);
                accessHandle.write(data, { at: 0 });
                accessHandle.flush();
                accessHandle.close();
                return { path, size: data.byteLength };
            } catch (e) {
                // Fall back to writable stream
            }
        }

        const writable = await fileHandle.createWritable();
        await writable.write(data);
        await writable.close();

        return { path, size: data.byteLength };
    }

    /**
     * Load data from a file
     * @param {string} path - File path
     * @returns {Promise<Uint8Array|null>}
     */
    async load(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Delete a file
     * @param {string} path - File path
     */
    async delete(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.removeEntry(fileName);
            return true;
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return false;
            }
            throw e;
        }
    }

    /**
     * List files in a directory
     * @param {string} dirPath - Directory path
     * @returns {Promise<string[]>} File names
     */
    async list(dirPath = '') {
        try {
            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const files = [];
            for await (const [name, handle] of dir.entries()) {
                files.push({
                    name,
                    type: handle.kind, // 'file' or 'directory'
                });
            }
            return files;
        } catch (e) {
            return [];
        }
    }

    /**
     * Check if a file exists
     * @param {string} path - File path
     */
    async exists(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.getFileHandle(fileName);
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Delete a directory and all contents
     * @param {string} dirPath - Directory path
     */
    async deleteDir(dirPath) {
        try {
            const parts = dirPath.split('/');
            const dirName = parts.pop();
            const parentPath = parts.join('/');

            const parent = parentPath ? await this.getDir(parentPath) : await this.getRoot();
            await parent.removeEntry(dirName, { recursive: true });
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Read a byte range from a file without loading the entire file
     * @param {string} path - File path
     * @param {number} offset - Start byte offset
     * @param {number} length - Number of bytes to read
     * @returns {Promise<Uint8Array|null>}
     */
    async readRange(path, offset, length) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();

            // Use slice to read only the requested range
            const blob = file.slice(offset, offset + length);
            const buffer = await blob.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Get file size without loading the file
     * @param {string} path - File path
     * @returns {Promise<number|null>}
     */
    async getFileSize(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();
            return file.size;
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Open a file for chunked reading
     * @param {string} path - File path
     * @returns {Promise<OPFSFileReader|null>}
     */
    async openFile(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            return new OPFSFileReader(fileHandle);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Check if OPFS is supported in this browser
     * @returns {Promise<boolean>}
     */
    async isSupported() {
        try {
            if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
                return false;
            }
            // Actually try to access OPFS
            await navigator.storage.getDirectory();
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Get storage statistics
     * @returns {Promise<{fileCount: number, totalSize: number}>}
     */
    async getStats() {
        try {
            const root = await this.getRoot();
            let fileCount = 0;
            let totalSize = 0;

            async function countDir(dir) {
                for await (const [name, handle] of dir.entries()) {
                    if (handle.kind === 'file') {
                        const file = await handle.getFile();
                        fileCount++;
                        totalSize += file.size;
                    } else if (handle.kind === 'directory') {
                        await countDir(handle);
                    }
                }
            }

            await countDir(root);
            return { fileCount, totalSize };
        } catch (e) {
            return { fileCount: 0, totalSize: 0 };
        }
    }

    /**
     * List all files in storage with their sizes
     * @returns {Promise<Array<{name: string, size: number, lastModified: number}>>}
     */
    async listFiles() {
        try {
            const root = await this.getRoot();
            const files = [];

            async function listDir(dir, prefix = '') {
                for await (const [name, handle] of dir.entries()) {
                    if (handle.kind === 'file') {
                        const file = await handle.getFile();
                        files.push({
                            name: prefix ? `${prefix}/${name}` : name,
                            size: file.size,
                            lastModified: file.lastModified
                        });
                    } else if (handle.kind === 'directory') {
                        await listDir(handle, prefix ? `${prefix}/${name}` : name);
                    }
                }
            }

            await listDir(root);
            return files;
        } catch (e) {
            return [];
        }
    }

    /**
     * Clear all files in storage
     * @returns {Promise<number>} Number of files deleted
     */
    async clearAll() {
        try {
            const root = await this.getRoot();
            let count = 0;

            const entries = [];
            for await (const [name, handle] of root.entries()) {
                entries.push({ name, kind: handle.kind });
            }

            for (const entry of entries) {
                await root.removeEntry(entry.name, { recursive: entry.kind === 'directory' });
                count++;
            }

            return count;
        } catch (e) {
            console.warn('Failed to clear OPFS:', e);
            return 0;
        }
    }
}

/**
 * OPFS File Reader for chunked/streaming reads
 * Wraps a FileSystemFileHandle for efficient byte-range access
 */
class OPFSFileReader {
    constructor(fileHandle) {
        this.fileHandle = fileHandle;
        this._file = null;
        this._size = null;
    }

    /**
     * Get the File object (cached)
     */
    async getFile() {
        if (!this._file) {
            this._file = await this.fileHandle.getFile();
            this._size = this._file.size;
        }
        return this._file;
    }

    /**
     * Get file size
     * @returns {Promise<number>}
     */
    async getSize() {
        if (this._size === null) {
            await this.getFile();
        }
        return this._size;
    }

    /**
     * Read a byte range
     * @param {number} offset - Start byte offset
     * @param {number} length - Number of bytes to read
     * @returns {Promise<Uint8Array>}
     */
    async readRange(offset, length) {
        const file = await this.getFile();
        const blob = file.slice(offset, offset + length);
        const buffer = await blob.arrayBuffer();
        return new Uint8Array(buffer);
    }

    /**
     * Read from end of file (useful for footer)
     * @param {number} length - Number of bytes to read from end
     * @returns {Promise<Uint8Array>}
     */
    async readFromEnd(length) {
        const size = await this.getSize();
        return this.readRange(size - length, length);
    }

    /**
     * Invalidate cache (call after file is modified)
     */
    invalidate() {
        this._file = null;
        this._size = null;
    }
}

/**
 * LRU Cache for page data
 * Keeps recently accessed pages in memory to avoid repeated OPFS reads
 */

export { OPFSStorage, OPFSFileReader };
