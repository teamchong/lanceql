/**
 * LanceQL Browser Client
 *
 * Modern SQL database in the browser with Lance columnar format,
 * vector search, and time travel support.
 *
 * @example Quick Start
 * ```js
 * import { vault } from '@anthropic/lanceql-browser';
 *
 * const v = await vault();
 * await v.exec('CREATE TABLE users (id INT, name TEXT)');
 * await v.exec("INSERT INTO users VALUES (1, 'Alice')");
 * const rows = await v.query('SELECT * FROM users');
 * ```
 */

// =============================================================================
// Primary API - Start here
// =============================================================================

// vault() - The main entry point. Creates an OPFS-backed SQL database.
export { Vault, TableRef, vault } from './store/vault.js';

// LocalDatabase - Lower-level API for direct table operations
export { LocalDatabase } from './database/local-database.js';

// =============================================================================
// Remote Data Access
// =============================================================================

// Read remote Lance files with HTTP Range requests
export { RemoteLanceFile } from './lance/remote-file.js';
export { RemoteLanceDataset } from './lance/remote-dataset.js';

// =============================================================================
// SQL Engine
// =============================================================================

export { SQLLexer } from './sql/lexer.js';
export { SQLParser } from './sql/parser.js';
export { SQLExecutor } from './sql/executor.js';

// =============================================================================
// Advanced: Storage & Internals
// =============================================================================

// Cache
export { MetadataCache } from './cache/metadata-cache.js';
export { LRUCache } from './cache/lru-cache.js';
export { HotTierCache, getHotTierCache } from './cache/hot-tier-cache.js';

// GPU Acceleration
export { WebGPUAccelerator, getWebGPUAccelerator } from './gpu/accelerator.js';
export { GPUAggregator, getGPUAggregator } from './gpu/aggregator.js';
export { GPUJoiner, getGPUJoiner } from './gpu/joiner.js';
export { GPUSorter, getGPUSorter } from './gpu/sorter.js';
export { GPUGrouper, getGPUGrouper } from './gpu/grouper.js';
export { GPUVectorSearch, getGPUVectorSearch } from './gpu/vector-search.js';

// Storage
export { OPFSStorage, OPFSFileReader } from './storage/opfs.js';
export { ChunkedLanceReader, MemoryManager } from './storage/lance-reader.js';
export { ProtobufEncoder, PureLanceWriter } from './storage/lance-writer.js';
export { DatasetStorage } from './storage/dataset-storage.js';

// Query Planning
export { StatisticsManager, CostModel, QueryPlanner } from './sql/query-planner.js';

// Lance Files
export { LanceFile } from './lance/lance-file.js';
export { LanceDataBase, OPFSLanceData, RemoteLanceData, DataFrame, LanceData, Statement, Database, openLance } from './lance/lance-data.js';
export { initSqlJs } from './lance/lance-data-sqljs.js';

// Vector Search
export { IVFIndex } from './search/ivf-index.js';

// Database (for remote datasets)
export { LanceDatabase } from './database/lance-database.js';
export { OPFSJoinExecutor } from './database/local-database.js';
export { MemoryTable, WorkerPool, SharedVectorStore } from './database/memory-table.js';

// Store (deprecated - use vault instead)
export { Store, lanceStore } from './store/store.js';

// WASM
export { LocalSQLParser, LanceFileWriter, wasmUtils, LanceQL } from './wasm/lanceql.js';

// =============================================================================
// Default Export - vault function for quick start
// =============================================================================

export { vault as default } from './store/vault.js';
