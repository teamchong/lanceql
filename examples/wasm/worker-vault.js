/**
 * WorkerVault - Unified storage with KV and SQL table support
 */

import { opfsStorage } from './opfs-storage.js';
import { encryptionKeys, importEncryptionKey, encryptData, decryptData } from './encryption.js';
import { WorkerDatabase } from './worker-database.js';
import { E, D } from './data-types.js';

class WorkerVault {
    constructor() {
        this._root = null;
        this._ready = false;
        this._kv = {};  // KV data loaded from JSON
        this._encryptionKeyId = null;
        this._db = null;  // Embedded database for SQL tables
    }

    async open(encryptionConfig = null) {
        if (this._ready) return this;

        // Set up encryption if provided
        if (encryptionConfig) {
            const { keyId, keyBytes } = encryptionConfig;
            if (!encryptionKeys.has(keyId)) {
                const cryptoKey = await importEncryptionKey(keyBytes);
                encryptionKeys.set(keyId, cryptoKey);
            }
            this._encryptionKeyId = keyId;
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._root = await opfsRoot.getDirectoryHandle('vault', { create: true });
            await this._loadKV();

            // Initialize embedded database for SQL tables
            this._db = new WorkerDatabase('vault');
            await this._db.open();

            this._ready = true;
        } catch (e) {
            console.error('[WorkerVault] Failed to open OPFS:', e);
            throw e;
        }

        return this;
    }

    _getCryptoKey() {
        return this._encryptionKeyId ? encryptionKeys.get(this._encryptionKeyId) : null;
    }

    async _loadKV() {
        try {
            const cryptoKey = this._getCryptoKey();
            const filename = cryptoKey ? '_vault.json.enc' : '_vault.json';
            const fileHandle = await this._root.getFileHandle(filename);
            const file = await fileHandle.getFile();

            if (cryptoKey) {
                const buffer = await file.arrayBuffer();
                this._kv = await decryptData(new Uint8Array(buffer), cryptoKey);
            } else {
                const text = await file.text();
                this._kv = JSON.parse(text);
            }
        } catch (e) {
            if (e.name === 'NotFoundError') {
                this._kv = {};
            } else {
                throw e;
            }
        }
    }

    async _saveKV() {
        const cryptoKey = this._getCryptoKey();
        const filename = cryptoKey ? '_vault.json.enc' : '_vault.json';
        const fileHandle = await this._root.getFileHandle(filename, { create: true });
        const writable = await fileHandle.createWritable();

        if (cryptoKey) {
            const encrypted = await encryptData(this._kv, cryptoKey);
            await writable.write(encrypted);
        } else {
            await writable.write(JSON.stringify(this._kv));
        }

        await writable.close();
    }

    async get(key) {
        return this._kv[key];
    }

    async set(key, value) {
        this._kv[key] = value;
        await this._saveKV();
    }

    async delete(key) {
        delete this._kv[key];
        await this._saveKV();
    }

    async keys() {
        return Object.keys(this._kv);
    }

    async has(key) {
        return key in this._kv;
    }

}

// Vault singleton
let vaultInstance = null;

async function getVault(encryptionConfig = null) {
    if (!vaultInstance) {
        vaultInstance = new WorkerVault();
    }
    // Re-open with encryption if not already done
    await vaultInstance.open(encryptionConfig);
    return vaultInstance;
}

export { WorkerVault };
