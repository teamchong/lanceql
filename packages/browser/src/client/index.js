/**
 * LanceQL Browser Client
 *
 * Modern SQL database in the browser with Lance columnar format,
 * vector search, and time travel support.
 *
 * @example Local Database
 * ```js
 * import { vault } from 'lanceql';
 *
 * const v = await vault();
 * await v.exec('CREATE TABLE users (id INT, name TEXT)');
 * await v.exec("INSERT INTO users VALUES (1, 'Alice')");
 * const result = await v.query('SELECT * FROM users');
 * ```
 *
 * @example Remote Dataset
 * ```js
 * import { LanceQL } from 'lanceql';
 *
 * const lanceql = await LanceQL.load();
 * const dataset = await lanceql.openDataset('https://example.com/data.lance');
 * const result = await dataset.executeSQL('SELECT * FROM data LIMIT 10');
 * ```
 *
 * @example Vector Search
 * ```js
 * const result = await dataset.executeSQL(`
 *   SELECT * FROM data
 *   WHERE embedding NEAR 'search query'
 *   LIMIT 20
 * `);
 * ```
 */

// =============================================================================
// Primary API
// =============================================================================

/**
 * Create an OPFS-backed SQL database in the browser.
 * This is the main entry point for local database operations.
 *
 * @example
 * const v = await vault();
 * await v.exec('CREATE TABLE users (id INT, name TEXT)');
 * const result = await v.query('SELECT * FROM users');
 */
export { vault, Vault, TableRef } from './store/vault.js';

/**
 * Load WASM module for remote dataset access.
 * Use this for querying remote Lance files with HTTP Range requests.
 *
 * @example
 * const lanceql = await LanceQL.load();
 * const dataset = await lanceql.openDataset('https://example.com/data.lance');
 * const result = await dataset.executeSQL('SELECT * FROM data LIMIT 10');
 */
export { LanceQL } from './wasm/lanceql.js';

// =============================================================================
// Advanced API (for power users)
// =============================================================================

/**
 * Direct access to remote Lance datasets.
 * Most users should use LanceQL.openDataset() instead.
 */
export { RemoteLanceDataset } from './lance/remote-dataset.js';

/**
 * Lower-level local database API.
 * Most users should use vault() instead.
 */
export { LocalDatabase } from './database/local-database.js';

/**
 * Create Lance files programmatically.
 * Useful for generating test data or exporting results.
 *
 * @example
 * const writer = new PureLanceWriter();
 * writer.addStringColumn('name', ['Alice', 'Bob']);
 * writer.addFloat32Column('embedding', new Float32Array([...]));
 * const bytes = writer.finalize();
 * await opfsStorage.save('data.lance', bytes);
 */
export { PureLanceWriter } from './storage/lance-writer.js';

/**
 * OPFS storage for Lance files.
 */
export { opfsStorage } from './storage/opfs.js';

// =============================================================================
// GPU Acceleration (WebGPU)
// =============================================================================

/**
 * GPU-accelerated operations for SQL and vector search.
 * Falls back to CPU if WebGPU is unavailable.
 */
export {
    getGPUAggregator,
    getGPUSorter,
    getGPUJoiner,
    getGPUGrouper,
    getGPUVectorSearch,
    DistanceMetric,
} from './webgpu/index.js';

// =============================================================================
// Default Export
// =============================================================================

export { vault as default } from './store/vault.js';
