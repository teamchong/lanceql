/**
 * Data Explorer - DevTools-style debugging UI
 *
 * Opens a popup window with:
 * - SQL editor
 * - DataFrame view
 * - DataView (virtualized table)
 * - Timeline (version history)
 *
 * @example
 * const v = await vault();
 * v.explorer(); // Opens debug window
 */

import { explorerHTML } from './explorer-html.js';

/**
 * Explorer controller - handles communication with popup window
 */
class ExplorerController {
    constructor(vault, explorerWindow, options) {
        this._vault = vault;
        this._window = explorerWindow;
        this._options = options;
        this._messageHandler = null;
        this._setupMessageHandler();
    }

    _setupMessageHandler() {
        this._messageHandler = async (event) => {
            // Only accept messages from our explorer window
            if (event.source !== this._window) return;

            const { type, id, ...params } = event.data || {};
            if (!type) return;

            try {
                let result;
                switch (type) {
                    case 'exec':
                        result = await this._vault.exec(params.sql);
                        break;
                    case 'query':
                        result = await this._vault.query(params.sql);
                        break;
                    case 'tables':
                        result = await this._vault.tables();
                        break;
                    case 'fetch-rows':
                        result = await this._fetchRows(params);
                        break;
                    case 'fetch-schema':
                        result = await this._fetchSchema(params.table);
                        break;
                    case 'fetch-count':
                        result = await this._fetchCount(params.table);
                        break;
                    case 'timeline':
                        result = await this._fetchTimeline(params.table);
                        break;
                    case 'diff':
                        result = await this._fetchDiff(params);
                        break;
                    case 'ready':
                        // Explorer window is ready, send initial data
                        this._sendInitialData();
                        return;
                    default:
                        console.warn('Unknown explorer message type:', type);
                        return;
                }

                this._window.postMessage({ type: 'response', id, result }, '*');
            } catch (error) {
                this._window.postMessage({
                    type: 'response',
                    id,
                    error: error.message
                }, '*');
            }
        };

        window.addEventListener('message', this._messageHandler);
    }

    async _fetchRows({ table, offset, limit, version }) {
        let sql = `SELECT * FROM ${table}`;
        if (version) sql += ` VERSION AS OF ${version}`;
        sql += ` LIMIT ${limit} OFFSET ${offset}`;
        return await this._vault.exec(sql);
    }

    async _fetchSchema(table) {
        const result = await this._vault.exec(`SELECT * FROM ${table} LIMIT 0`);
        return { columns: result?.columns || [] };
    }

    async _fetchCount(table) {
        const result = await this._vault.exec(`SELECT COUNT(*) FROM ${table}`);
        return result?.rows?.[0]?.[0] || 0;
    }

    async _fetchTimeline(table) {
        // Use SHOW VERSIONS if available, otherwise return mock data
        try {
            const result = await this._vault.exec(`SHOW VERSIONS FOR ${table}`);
            return result?.rows || [];
        } catch {
            // Fallback - return current version only
            return [{ version: 1, timestamp: Date.now(), operation: 'CURRENT', rowCount: 0, delta: '+0' }];
        }
    }

    async _fetchDiff({ table, from, to }) {
        try {
            const result = await this._vault.exec(
                `DIFF ${table} VERSION ${from} AND VERSION ${to}`
            );
            return result;
        } catch (e) {
            return { added: [], deleted: [], error: e.message };
        }
    }

    async _sendInitialData() {
        const tables = await this._vault.tables();
        this._window.postMessage({
            type: 'init',
            tables,
            options: this._options
        }, '*');
    }

    /**
     * Run SQL in the explorer
     */
    runSQL(sql) {
        this._window.postMessage({ type: 'run-sql', sql }, '*');
    }

    /**
     * Select a table in the explorer
     */
    selectTable(table) {
        this._window.postMessage({ type: 'select-table', table }, '*');
    }

    /**
     * Close the explorer window
     */
    close() {
        window.removeEventListener('message', this._messageHandler);
        if (this._window && !this._window.closed) {
            this._window.close();
        }
    }
}

/**
 * Open the Data Explorer in a new window
 *
 * @param {Vault} vault - Vault instance
 * @param {Object} options - Explorer options
 * @param {string} [options.table] - Pre-select table
 * @param {number} [options.version] - Start at specific version
 * @param {number} [options.width=1200] - Window width
 * @param {number} [options.height=800] - Window height
 * @returns {ExplorerController} Controller for the explorer window
 */
export function openExplorer(vault, options = {}) {
    const {
        width = 1200,
        height = 800,
        table = null,
        version = null
    } = options;

    // Calculate window position (centered)
    const left = (screen.width - width) / 2;
    const top = (screen.height - height) / 2;

    // Open popup window
    const explorerWindow = window.open(
        '',
        'lanceql-explorer',
        `width=${width},height=${height},left=${left},top=${top},menubar=no,toolbar=no,location=no,status=no`
    );

    if (!explorerWindow) {
        throw new Error('Failed to open explorer window. Check popup blocker settings.');
    }

    // Write the explorer HTML
    explorerWindow.document.write(explorerHTML);
    explorerWindow.document.close();

    // Create controller
    return new ExplorerController(vault, explorerWindow, { table, version });
}
