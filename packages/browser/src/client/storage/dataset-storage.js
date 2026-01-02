/**
 * DatasetStorage - Legacy IndexedDB + OPFS storage management
 */

import { OPFSStorage } from './opfs.js';

class DatasetStorage {
    constructor(dbName = 'lanceql-files', version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
        this.SIZE_THRESHOLD = 50 * 1024 * 1024; // 50MB
    }

    async open() {
        if (this.db) return this.db;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('files')) {
                    db.createObjectStore('files', { keyPath: 'name' });
                }
                if (!db.objectStoreNames.contains('index')) {
                    const store = db.createObjectStore('index', { keyPath: 'name' });
                    store.createIndex('timestamp', 'timestamp');
                    store.createIndex('size', 'size');
                }
            };
        });
    }

    async hasOPFS() {
        try {
            return 'storage' in navigator && 'getDirectory' in navigator.storage;
        } catch {
            return false;
        }
    }

    async save(name, data, metadata = {}) {
        const db = await this.open();
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const size = bytes.byteLength;
        const useOPFS = size >= this.SIZE_THRESHOLD && await this.hasOPFS();

        if (useOPFS) {
            try {
                const root = await navigator.storage.getDirectory();
                const fileHandle = await root.getFileHandle(name, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(bytes);
                await writable.close();
            } catch (e) {
                console.warn('[DatasetStorage] OPFS save failed, falling back to IndexedDB:', e);
            }
        }

        if (!useOPFS) {
            await new Promise((resolve, reject) => {
                const tx = db.transaction('files', 'readwrite');
                const store = tx.objectStore('files');
                store.put({ name, data: bytes });
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
        }

        await new Promise((resolve, reject) => {
            const tx = db.transaction('index', 'readwrite');
            const store = tx.objectStore('index');
            store.put({
                name,
                size,
                timestamp: Date.now(),
                storage: useOPFS ? 'opfs' : 'indexeddb',
                ...metadata
            });
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });

        return { name, size, storage: useOPFS ? 'opfs' : 'indexeddb' };
    }

    async load(name) {
        const db = await this.open();

        const entry = await new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
        });

        if (!entry) return null;

        if (entry.storage === 'opfs') {
            try {
                const root = await navigator.storage.getDirectory();
                const fileHandle = await root.getFileHandle(name);
                const file = await fileHandle.getFile();
                const buffer = await file.arrayBuffer();
                return new Uint8Array(buffer);
            } catch (e) {
                console.warn('[DatasetStorage] OPFS load failed:', e);
                return null;
            }
        }

        return new Promise((resolve) => {
            const tx = db.transaction('files', 'readonly');
            const store = tx.objectStore('files');
            const request = store.get(name);
            request.onsuccess = () => {
                const result = request.result;
                resolve(result ? result.data : null);
            };
            request.onerror = () => resolve(null);
        });
    }

    async list() {
        const db = await this.open();

        return new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => resolve([]);
        });
    }

    async delete(name) {
        const db = await this.open();

        const entry = await new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
        });

        if (entry?.storage === 'opfs') {
            try {
                const root = await navigator.storage.getDirectory();
                await root.removeEntry(name);
            } catch (e) {
                console.warn('[DatasetStorage] OPFS delete failed:', e);
            }
        }

        await new Promise((resolve) => {
            const tx = db.transaction('files', 'readwrite');
            const store = tx.objectStore('files');
            store.delete(name);
            tx.oncomplete = () => resolve();
        });

        // Delete from index
        await new Promise((resolve) => {
            const tx = db.transaction('index', 'readwrite');
            const store = tx.objectStore('index');
            store.delete(name);
            tx.oncomplete = () => resolve();
        });
    }

    /**
     * Check if a dataset exists.
     * @param {string} name - Dataset name
     * @returns {Promise<boolean>}
     */
    async exists(name) {
        const db = await this.open();

        return new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(!!request.result);
            request.onerror = () => resolve(false);
        });
    }

    /**
     * Get storage usage info.
     * @returns {Promise<Object>} Usage stats
     */
    async getUsage() {
        const datasets = await this.list();
        const totalSize = datasets.reduce((sum, d) => sum + (d.size || 0), 0);
        const indexedDBCount = datasets.filter(d => d.storage === 'indexeddb').length;
        const opfsCount = datasets.filter(d => d.storage === 'opfs').length;

        let quota = null;
        if (navigator.storage?.estimate) {
            quota = await navigator.storage.estimate();
        }

        return {
            datasets: datasets.length,
            totalSize,
            indexedDBCount,
            opfsCount,
            quota
        };
    }
}

// Global storage instances
const opfsStorage = new OPFSStorage();  // OPFS-only (recommended)
const datasetStorage = new DatasetStorage();  // Legacy IndexedDB + OPFS


export { DatasetStorage };
