/**
 * LanceQL Browser Client
 *
 * Modular client library for Lance columnar file format, vector search,
 * and SQL query execution in the browser.
 */

// Cache
export { MetadataCache } from './cache/metadata-cache.js';
export { LRUCache } from './cache/lru-cache.js';
export { HotTierCache, hotTierCache } from './cache/hot-tier-cache.js';

// GPU Acceleration
export { WebGPUAccelerator, webgpuAccelerator } from './gpu/accelerator.js';
export { GPUAggregator } from './gpu/aggregator.js';
export { GPUJoiner } from './gpu/joiner.js';
export { GPUSorter } from './gpu/sorter.js';
export { GPUGrouper } from './gpu/grouper.js';
export { GPUVectorSearch } from './gpu/vector-search.js';

// Storage
export { OPFSStorage, OPFSFileReader } from './storage/opfs.js';
export { ChunkedLanceReader, MemoryManager } from './storage/lance-reader.js';
export { ProtobufEncoder, PureLanceWriter } from './storage/lance-writer.js';
export { DatasetStorage } from './storage/dataset-storage.js';

// SQL
export { SQLLexer } from './sql/lexer.js';
export { SQLParser } from './sql/parser.js';
export { SQLExecutor } from './sql/executor.js';
export { StatisticsManager, CostModel, QueryPlanner } from './sql/query-planner.js';

// Lance Files
export { LanceFile } from './lance/lance-file.js';
export { RemoteLanceFile } from './lance/remote-file.js';
export { RemoteLanceDataset } from './lance/remote-dataset.js';
export { LanceDataBase, OPFSLanceData, RemoteLanceData, DataFrame, LanceData, Statement, Database } from './lance/lance-data.js';

// Vector Search
export { IVFIndex } from './search/ivf-index.js';

// Database
export { LanceDatabase } from './database/lance-database.js';
export { OPFSJoinExecutor, LocalDatabase } from './database/local-database.js';
export { MemoryTable, WorkerPool, SharedVectorStore } from './database/memory-table.js';

// Store
export { Store, lanceStore } from './store/store.js';
export { Vault, TableRef, vault } from './store/vault.js';

// WASM
export { LocalSQLParser, LanceFileWriter, wasmUtils, LanceQL } from './wasm/lanceql.js';

// Aliases for backwards compatibility
export { lanceStore as createStore } from './store/store.js';
export { Store as KeyValueStore } from './store/store.js';
export { Database as SqlJsDatabase, Statement as SqlJsStatement } from './lance/lance-data.js';

// Default export
export { LanceQL as default } from './wasm/lanceql.js';

// TODO: Add singleton instances when needed
// export const metadataCache = new MetadataCache();
// export const webgpuAccelerator = new WebGPUAccelerator();
// etc.
