/**
 * LanceQL Vue Bindings
 *
 * Vue 3 composables for the LanceQL Store API.
 * Provides reactive state management with OPFS persistence.
 *
 * @example
 * <script setup>
 * import { useStore, useQuery, useSearch } from '@metal0/lanceql/vue';
 *
 * const { data: user, set, loading } = useStore('myapp', 'user');
 * const { data: products } = useQuery('myapp', 'products', { price: { $lt: 100 } });
 * const { results, search, query } = useSearch('myapp', 'products');
 * </script>
 *
 * <template>
 *   <input v-model="query" @input="search(query)" placeholder="Search..." />
 *   <div v-for="r in results" :key="r.item.id">{{ r.item.name }}</div>
 * </template>
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
 * Check if Vue is available and get its reactivity functions.
 */
function getVue() {
    if (typeof window !== 'undefined' && window.Vue) {
        return window.Vue;
    }
    throw new Error('Vue 3 is required. Make sure Vue is loaded before using these composables.');
}

/**
 * useStore - Vue composable for key-value storage.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Key to read/write
 * @param {Object} options - Composable options
 * @returns {Object} { data, set, remove, loading, error }
 *
 * @example
 * <script setup>
 * import { useStore } from '@metal0/lanceql/vue';
 *
 * const { data: user, set, loading } = useStore('myapp', 'user');
 *
 * function updateName(name) {
 *   set({ ...user.value, name });
 * }
 * </script>
 */
export function useStore(storeName, key, options = {}) {
    const Vue = getVue();
    const { ref, onMounted, onUnmounted } = Vue;

    const data = ref(options.initialValue);
    const loading = ref(true);
    const error = ref(null);
    let store = null;

    onMounted(async () => {
        try {
            store = await getStore(storeName, options.storeOptions);
            const value = await store.get(key);
            data.value = value;
            loading.value = false;
        } catch (e) {
            error.value = e;
            loading.value = false;
        }
    });

    async function set(value) {
        try {
            loading.value = true;
            if (!store) store = await getStore(storeName, options.storeOptions);
            await store.set(key, value);
            data.value = value;
            loading.value = false;
        } catch (e) {
            error.value = e;
            loading.value = false;
        }
    }

    async function remove() {
        try {
            if (!store) store = await getStore(storeName, options.storeOptions);
            await store.delete(key);
            data.value = undefined;
        } catch (e) {
            error.value = e;
        }
    }

    return { data, set, remove, loading, error };
}

/**
 * useQuery - Vue composable for filtered queries.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object|Ref} query - Filter query (can be reactive)
 * @param {Object} options - Composable options
 * @returns {Object} { data, loading, error, refetch }
 *
 * @example
 * <script setup>
 * import { useQuery } from '@metal0/lanceql/vue';
 * import { ref, computed } from 'vue';
 *
 * const maxPrice = ref(100);
 * const query = computed(() => ({ price: { $lt: maxPrice.value } }));
 * const { data: products, loading } = useQuery('myapp', 'products', query);
 * </script>
 */
export function useQuery(storeName, key, query = {}, options = {}) {
    const Vue = getVue();
    const { ref, watch, onMounted, isRef, unref } = Vue;

    const data = ref([]);
    const loading = ref(true);
    const error = ref(null);

    async function fetch() {
        try {
            loading.value = true;
            const store = await getStore(storeName, options.storeOptions);
            const q = isRef(query) ? unref(query) : query;
            const results = await store.filter(key, q);
            data.value = results;
            loading.value = false;
        } catch (e) {
            error.value = e;
            loading.value = false;
        }
    }

    onMounted(fetch);

    // Watch for query changes if it's reactive
    if (isRef(query)) {
        watch(query, fetch, { deep: true });
    }

    return { data, loading, error, refetch: fetch };
}

/**
 * useSearch - Vue composable for semantic search.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} options - Search options
 * @returns {Object} { results, query, search, loading, error, clear }
 *
 * @example
 * <script setup>
 * import { useSearch } from '@metal0/lanceql/vue';
 *
 * const { results, query, search, loading } = useSearch('myapp', 'products', {
 *   limit: 5,
 *   debounce: 300
 * });
 * </script>
 *
 * <template>
 *   <input v-model="query" @input="search(query)" placeholder="Search..." />
 *   <div v-for="r in results" :key="r.item.id">
 *     {{ r.item.name }} ({{ r.score.toFixed(2) }})
 *   </div>
 * </template>
 */
export function useSearch(storeName, key, options = {}) {
    const Vue = getVue();
    const { ref, onUnmounted } = Vue;

    const limit = options.limit || 10;
    const debounceMs = options.debounce || 300;

    const results = ref([]);
    const query = ref('');
    const loading = ref(false);
    const error = ref(null);
    let debounceTimer = null;

    async function search(text) {
        query.value = text;

        if (debounceTimer) clearTimeout(debounceTimer);

        if (!text || text.trim().length === 0) {
            results.value = [];
            loading.value = false;
            return;
        }

        loading.value = true;

        debounceTimer = setTimeout(async () => {
            try {
                const store = await getStore(storeName, options.storeOptions);
                const searchResults = await store.search(key, text, limit);
                results.value = searchResults;
                loading.value = false;
            } catch (e) {
                error.value = e;
                loading.value = false;
            }
        }, debounceMs);
    }

    function clear() {
        if (debounceTimer) clearTimeout(debounceTimer);
        query.value = '';
        results.value = [];
        loading.value = false;
    }

    onUnmounted(() => {
        if (debounceTimer) clearTimeout(debounceTimer);
    });

    return { results, query, search, loading, error, clear };
}

/**
 * useCollection - Vue composable for managing a collection with CRUD.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @returns {Object} { items, add, remove, update, clear, loading, error }
 *
 * @example
 * <script setup>
 * import { useCollection } from '@metal0/lanceql/vue';
 *
 * const { items: todos, add, remove, update } = useCollection('myapp', 'todos');
 *
 * async function addTodo(text) {
 *   await add({ id: Date.now(), text, done: false });
 * }
 *
 * async function toggleTodo(id) {
 *   const todo = todos.value.find(t => t.id === id);
 *   await update(id, { done: !todo.done });
 * }
 * </script>
 */
export function useCollection(storeName, key, options = {}) {
    const Vue = getVue();
    const { ref, onMounted } = Vue;

    const items = ref([]);
    const loading = ref(true);
    const error = ref(null);
    let store = null;

    async function load() {
        try {
            loading.value = true;
            store = await getStore(storeName, options.storeOptions);
            const data = await store.get(key);
            items.value = data || [];
            loading.value = false;
        } catch (e) {
            error.value = e;
            loading.value = false;
        }
    }

    async function save() {
        try {
            if (!store) store = await getStore(storeName, options.storeOptions);
            await store.set(key, items.value);
        } catch (e) {
            error.value = e;
        }
    }

    async function add(item) {
        items.value = [...items.value, item];
        await save();
    }

    async function remove(predicate) {
        if (typeof predicate === 'function') {
            items.value = items.value.filter(item => !predicate(item));
        } else {
            const id = typeof predicate === 'object' ? predicate.id : predicate;
            items.value = items.value.filter(item => item.id !== id);
        }
        await save();
    }

    async function update(id, updates) {
        items.value = items.value.map(item =>
            item.id === id ? { ...item, ...updates } : item
        );
        await save();
    }

    async function clear() {
        items.value = [];
        await save();
    }

    onMounted(load);

    return { items, add, remove, update, clear, reload: load, loading, error };
}

/**
 * useStoreKeys - Vue composable to list all keys in a store.
 *
 * @param {string} storeName - Store name
 * @returns {Object} { keys, loading, error }
 */
export function useStoreKeys(storeName, options = {}) {
    const Vue = getVue();
    const { ref, onMounted } = Vue;

    const keys = ref([]);
    const loading = ref(true);
    const error = ref(null);

    onMounted(async () => {
        try {
            const store = await getStore(storeName, options.storeOptions);
            const allKeys = await store.keys();
            keys.value = allKeys;
            loading.value = false;
        } catch (e) {
            error.value = e;
            loading.value = false;
        }
    });

    return { keys, loading, error };
}

/**
 * useSemanticSearch - Vue composable for WebGPU-accelerated semantic search.
 *
 * Automatically loads the AI model and enables GPU-accelerated text encoding.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} options - Search options
 * @param {string} options.model - Model name ('minilm', 'clip', or GGUF URL)
 * @param {number} options.limit - Max results (default 10)
 * @param {number} options.debounce - Debounce delay in ms (default 300)
 * @returns {Object} { results, query, search, loading, modelLoading, modelInfo, error, clear }
 *
 * @example
 * <script setup>
 * import { useSemanticSearch } from '@metal0/lanceql/vue';
 *
 * const {
 *   results, query, search, loading, modelLoading, modelInfo
 * } = useSemanticSearch('myapp', 'products', {
 *   model: 'minilm',
 *   limit: 5
 * });
 * </script>
 *
 * <template>
 *   <div v-if="modelLoading">Loading AI model...</div>
 *   <template v-else>
 *     <p>Model: {{ modelInfo?.model }} ({{ modelInfo?.dimensions }}D)</p>
 *     <input v-model="query" @input="search(query)" placeholder="AI-powered search..." />
 *     <div v-if="loading">Searching...</div>
 *     <div v-for="r in results" :key="r.item.id">
 *       {{ r.item.name }} ({{ (r.score * 100).toFixed(0) }}%)
 *     </div>
 *   </template>
 * </template>
 */
export function useSemanticSearch(storeName, key, options = {}) {
    const Vue = getVue();
    const { ref, onMounted, onUnmounted } = Vue;

    const model = options.model || 'minilm';
    const limit = options.limit || 10;
    const debounceMs = options.debounce || 300;

    const results = ref([]);
    const query = ref('');
    const loading = ref(false);
    const modelLoading = ref(true);
    const modelInfo = ref(null);
    const error = ref(null);
    let debounceTimer = null;
    let store = null;

    onMounted(async () => {
        try {
            store = await getStore(storeName, options.storeOptions);

            // Enable semantic search if not already enabled
            if (!store.hasSemanticSearch()) {
                const info = await store.enableSemanticSearch({
                    model,
                    onProgress: options.onModelProgress
                });
                modelInfo.value = info;
            } else {
                modelInfo.value = {
                    model: store._embedder?.model,
                    dimensions: store._embedder?.dimensions
                };
            }
            modelLoading.value = false;
        } catch (e) {
            error.value = e;
            modelLoading.value = false;
        }
    });

    async function search(text) {
        query.value = text;

        if (debounceTimer) clearTimeout(debounceTimer);

        if (!text || text.trim().length === 0) {
            results.value = [];
            loading.value = false;
            return;
        }

        loading.value = true;

        debounceTimer = setTimeout(async () => {
            try {
                if (!store) store = await getStore(storeName, options.storeOptions);
                const searchResults = await store.search(key, text, limit);
                results.value = searchResults;
                loading.value = false;
            } catch (e) {
                error.value = e;
                loading.value = false;
            }
        }, debounceMs);
    }

    function clear() {
        if (debounceTimer) clearTimeout(debounceTimer);
        query.value = '';
        results.value = [];
        loading.value = false;
    }

    onUnmounted(() => {
        if (debounceTimer) clearTimeout(debounceTimer);
    });

    return { results, query, search, loading, modelLoading, modelInfo, error, clear };
}

/**
 * useGPUInfo - Vue composable to get WebGPU device information.
 *
 * @returns {Object} { info, loading, available }
 *
 * @example
 * <script setup>
 * import { useGPUInfo } from '@metal0/lanceql/vue';
 *
 * const { info, loading, available } = useGPUInfo();
 * </script>
 *
 * <template>
 *   <div v-if="loading">Checking GPU...</div>
 *   <div v-else-if="available">
 *     <p>GPU: {{ info.device }}</p>
 *     <p>Vendor: {{ info.vendor }}</p>
 *   </div>
 *   <div v-else>WebGPU not available</div>
 * </template>
 */
export function useGPUInfo() {
    const Vue = getVue();
    const { ref, onMounted } = Vue;

    const info = ref(null);
    const loading = ref(true);
    const available = ref(false);

    onMounted(async () => {
        try {
            const webgpu = await import('./webgpu/index.js');

            if (!webgpu.isWebGPUAvailable()) {
                available.value = false;
                loading.value = false;
                return;
            }

            const gpuInfo = await webgpu.getWebGPUInfo();
            info.value = gpuInfo;
            available.value = gpuInfo !== null;
            loading.value = false;
        } catch (e) {
            available.value = false;
            loading.value = false;
        }
    });

    return { info, loading, available };
}
