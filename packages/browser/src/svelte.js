/**
 * LanceQL Svelte Bindings
 *
 * Svelte stores for the LanceQL Store API.
 * Provides reactive state management with OPFS persistence.
 *
 * @example
 * <script>
 *   import { createLanceStore, createQueryStore, createSearchStore } from '@metal0/lanceql/svelte';
 *
 *   const user = createLanceStore('myapp', 'user');
 *   const products = createQueryStore('myapp', 'products', { price: { $lt: 100 } });
 *   const results = createSearchStore('myapp', 'products');
 * </script>
 *
 * <input bind:value={$results.query} placeholder="Search..." />
 * {#each $results.items as result}
 *   <div>{result.item.name}</div>
 * {/each}
 */

import { lanceStore } from './lanceql.js';

// Global store cache
const storeCache = new Map();

async function getStore(name, options = {}) {
    const key = `${name}:${JSON.stringify(options)}`;
    if (!storeCache.has(key)) {
        storeCache.set(key, lanceStore(name, options));
    }
    return storeCache.get(key);
}

/**
 * Create a Svelte-compatible writable store interface.
 * Works with Svelte's $ syntax for auto-subscription.
 */
function createWritableStore(initialValue, start = () => {}) {
    let value = initialValue;
    const subscribers = new Set();

    function set(newValue) {
        value = newValue;
        subscribers.forEach(fn => fn(value));
    }

    function update(fn) {
        set(fn(value));
    }

    function subscribe(fn) {
        subscribers.add(fn);
        fn(value);
        const stop = start(set);
        return () => {
            subscribers.delete(fn);
            if (subscribers.size === 0 && stop) stop();
        };
    }

    return { subscribe, set, update };
}

/**
 * createLanceStore - Create a Svelte store backed by LanceQL.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Key to store
 * @param {any} initialValue - Initial value while loading
 * @returns {Object} Svelte store { subscribe, set, update }
 *
 * @example
 * <script>
 *   import { createLanceStore } from '@metal0/lanceql/svelte';
 *
 *   const user = createLanceStore('myapp', 'user', null);
 *
 *   async function updateName(name) {
 *     user.set({ ...$user, name });
 *   }
 * </script>
 *
 * {#if $user}
 *   <h1>{$user.name}</h1>
 * {:else}
 *   <p>Loading...</p>
 * {/if}
 */
export function createLanceStore(storeName, key, initialValue = undefined) {
    let store = null;
    let loaded = false;

    const { subscribe, set, update } = createWritableStore(
        { value: initialValue, loading: true, error: null },
        (setStore) => {
            // Load initial value
            (async () => {
                try {
                    store = await getStore(storeName);
                    const value = await store.get(key);
                    loaded = true;
                    setStore({ value, loading: false, error: null });
                } catch (e) {
                    setStore({ value: initialValue, loading: false, error: e });
                }
            })();
        }
    );

    // Custom set that persists to LanceQL
    async function setValue(newValue) {
        try {
            if (!store) store = await getStore(storeName);
            await store.set(key, newValue);
            set({ value: newValue, loading: false, error: null });
        } catch (e) {
            update(s => ({ ...s, error: e }));
        }
    }

    // Custom update that persists to LanceQL
    async function updateValue(fn) {
        let currentValue;
        subscribe(s => { currentValue = s.value; })();
        const newValue = fn(currentValue);
        await setValue(newValue);
    }

    return {
        subscribe,
        set: setValue,
        update: updateValue
    };
}

/**
 * createQueryStore - Create a Svelte store for filtered queries.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} query - Filter query
 * @returns {Object} Svelte store { subscribe, refetch, setQuery }
 *
 * @example
 * <script>
 *   import { createQueryStore } from '@metal0/lanceql/svelte';
 *
 *   const products = createQueryStore('myapp', 'products', { price: { $lt: 100 } });
 * </script>
 *
 * {#if $products.loading}
 *   <p>Loading...</p>
 * {:else}
 *   {#each $products.items as product}
 *     <div>{product.name}</div>
 *   {/each}
 * {/if}
 */
export function createQueryStore(storeName, key, initialQuery = {}) {
    let currentQuery = initialQuery;

    const { subscribe, set } = createWritableStore(
        { items: [], loading: true, error: null, query: initialQuery }
    );

    async function fetch(query = currentQuery) {
        currentQuery = query;
        set({ items: [], loading: true, error: null, query });

        try {
            const store = await getStore(storeName);
            const items = await store.filter(key, query);
            set({ items, loading: false, error: null, query });
        } catch (e) {
            set({ items: [], loading: false, error: e, query });
        }
    }

    // Initial fetch
    fetch();

    return {
        subscribe,
        refetch: () => fetch(currentQuery),
        setQuery: (newQuery) => fetch(newQuery)
    };
}

/**
 * createSearchStore - Create a Svelte store for semantic search.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} options - Search options
 * @returns {Object} Svelte store with search functionality
 *
 * @example
 * <script>
 *   import { createSearchStore } from '@metal0/lanceql/svelte';
 *
 *   const search = createSearchStore('myapp', 'products', { limit: 5 });
 * </script>
 *
 * <input
 *   value={$search.query}
 *   on:input={e => search.search(e.target.value)}
 *   placeholder="Search products..."
 * />
 *
 * {#each $search.results as result}
 *   <div>{result.item.name} ({result.score.toFixed(2)})</div>
 * {/each}
 */
export function createSearchStore(storeName, key, options = {}) {
    const limit = options.limit || 10;
    const debounce = options.debounce || 300;
    let debounceTimer = null;

    const { subscribe, set, update } = createWritableStore({
        query: '',
        results: [],
        loading: false,
        error: null
    });

    async function search(text) {
        update(s => ({ ...s, query: text }));

        if (debounceTimer) clearTimeout(debounceTimer);

        if (!text || text.trim().length === 0) {
            set({ query: text, results: [], loading: false, error: null });
            return;
        }

        update(s => ({ ...s, loading: true }));

        debounceTimer = setTimeout(async () => {
            try {
                const store = await getStore(storeName);
                const results = await store.search(key, text, limit);
                set({ query: text, results, loading: false, error: null });
            } catch (e) {
                update(s => ({ ...s, loading: false, error: e }));
            }
        }, debounce);
    }

    function clear() {
        if (debounceTimer) clearTimeout(debounceTimer);
        set({ query: '', results: [], loading: false, error: null });
    }

    return {
        subscribe,
        search,
        clear
    };
}

/**
 * createCollectionStore - Create a Svelte store for managing a collection.
 * Supports CRUD operations with automatic persistence.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @returns {Object} Svelte store with CRUD methods
 *
 * @example
 * <script>
 *   import { createCollectionStore } from '@metal0/lanceql/svelte';
 *
 *   const todos = createCollectionStore('myapp', 'todos');
 *
 *   async function addTodo(text) {
 *     await todos.add({ id: Date.now(), text, done: false });
 *   }
 * </script>
 */
export function createCollectionStore(storeName, key) {
    let store = null;
    let items = [];

    const { subscribe, set } = createWritableStore({
        items: [],
        loading: true,
        error: null
    });

    async function load() {
        try {
            store = await getStore(storeName);
            items = await store.get(key) || [];
            set({ items, loading: false, error: null });
        } catch (e) {
            set({ items: [], loading: false, error: e });
        }
    }

    async function save() {
        try {
            if (!store) store = await getStore(storeName);
            await store.set(key, items);
            set({ items, loading: false, error: null });
        } catch (e) {
            set({ items, loading: false, error: e });
        }
    }

    async function add(item) {
        items = [...items, item];
        await save();
    }

    async function remove(predicate) {
        if (typeof predicate === 'function') {
            items = items.filter(item => !predicate(item));
        } else {
            // Assume it's an ID or object with id
            const id = typeof predicate === 'object' ? predicate.id : predicate;
            items = items.filter(item => item.id !== id);
        }
        await save();
    }

    async function update(id, updates) {
        items = items.map(item =>
            item.id === id ? { ...item, ...updates } : item
        );
        await save();
    }

    async function clear() {
        items = [];
        await save();
    }

    // Initial load
    load();

    return {
        subscribe,
        add,
        remove,
        update,
        clear,
        reload: load
    };
}

/**
 * createSemanticSearchStore - Create a Svelte store for WebGPU-accelerated semantic search.
 *
 * Automatically loads the AI model and enables GPU-accelerated text encoding.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} options - Search options
 * @param {string} options.model - Model name ('minilm', 'clip', or GGUF URL)
 * @param {number} options.limit - Max results (default 10)
 * @param {number} options.debounce - Debounce delay in ms (default 300)
 * @returns {Object} Svelte store with semantic search functionality
 *
 * @example
 * <script>
 *   import { createSemanticSearchStore } from '@metal0/lanceql/svelte';
 *
 *   const search = createSemanticSearchStore('myapp', 'products', {
 *     model: 'minilm',
 *     limit: 5
 *   });
 * </script>
 *
 * {#if $search.modelLoading}
 *   <p>Loading AI model...</p>
 * {:else}
 *   <p>Model: {$search.modelInfo?.model} ({$search.modelInfo?.dimensions}D)</p>
 *   <input
 *     value={$search.query}
 *     on:input={e => search.search(e.target.value)}
 *     placeholder="AI-powered search..."
 *   />
 *   {#each $search.results as result}
 *     <div>{result.item.name} ({(result.score * 100).toFixed(0)}%)</div>
 *   {/each}
 * {/if}
 */
export function createSemanticSearchStore(storeName, key, options = {}) {
    const model = options.model || 'minilm';
    const limit = options.limit || 10;
    const debounce = options.debounce || 300;
    let debounceTimer = null;
    let store = null;

    const { subscribe, set, update } = createWritableStore({
        query: '',
        results: [],
        loading: false,
        modelLoading: true,
        modelInfo: null,
        error: null
    });

    // Initialize semantic search
    (async () => {
        try {
            store = await getStore(storeName);

            // Enable semantic search if not already enabled
            if (!store.hasSemanticSearch()) {
                const info = await store.enableSemanticSearch({
                    model,
                    onProgress: options.onModelProgress
                });

                update(s => ({ ...s, modelLoading: false, modelInfo: info }));
            } else {
                update(s => ({
                    ...s,
                    modelLoading: false,
                    modelInfo: {
                        model: store._embedder?.model,
                        dimensions: store._embedder?.dimensions
                    }
                }));
            }
        } catch (e) {
            update(s => ({ ...s, modelLoading: false, error: e }));
        }
    })();

    async function search(text) {
        update(s => ({ ...s, query: text }));

        if (debounceTimer) clearTimeout(debounceTimer);

        if (!text || text.trim().length === 0) {
            update(s => ({ ...s, results: [], loading: false }));
            return;
        }

        update(s => ({ ...s, loading: true }));

        debounceTimer = setTimeout(async () => {
            try {
                if (!store) store = await getStore(storeName);
                const results = await store.search(key, text, limit);
                update(s => ({ ...s, results, loading: false }));
            } catch (e) {
                update(s => ({ ...s, loading: false, error: e }));
            }
        }, debounce);
    }

    function clear() {
        if (debounceTimer) clearTimeout(debounceTimer);
        update(s => ({ ...s, query: '', results: [], loading: false }));
    }

    return {
        subscribe,
        search,
        clear
    };
}

/**
 * createGPUInfoStore - Create a Svelte store for WebGPU device information.
 *
 * @returns {Object} Svelte store { subscribe } with GPU info
 *
 * @example
 * <script>
 *   import { createGPUInfoStore } from '@metal0/lanceql/svelte';
 *
 *   const gpu = createGPUInfoStore();
 * </script>
 *
 * {#if $gpu.loading}
 *   <p>Checking GPU...</p>
 * {:else if $gpu.available}
 *   <p>GPU: {$gpu.info.device}</p>
 *   <p>Vendor: {$gpu.info.vendor}</p>
 * {:else}
 *   <p>WebGPU not available</p>
 * {/if}
 */
export function createGPUInfoStore() {
    const { subscribe, set } = createWritableStore({
        info: null,
        loading: true,
        available: false
    });

    // Check WebGPU availability
    (async () => {
        try {
            const webgpu = await import('./webgpu/index.js');

            if (!webgpu.isWebGPUAvailable()) {
                set({ info: null, loading: false, available: false });
                return;
            }

            const gpuInfo = await webgpu.getWebGPUInfo();
            set({
                info: gpuInfo,
                loading: false,
                available: gpuInfo !== null
            });
        } catch (e) {
            set({ info: null, loading: false, available: false });
        }
    })();

    return { subscribe };
}
