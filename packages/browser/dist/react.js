/**
 * LanceQL React Bindings
 *
 * React hooks for the LanceQL Store API.
 * Provides reactive state management with OPFS persistence.
 *
 * @example
 * import { useStore, useQuery, useSearch } from '@metal0/lanceql/react';
 *
 * function App() {
 *   const { data: user, set, loading } = useStore('user');
 *   const products = useQuery('products', { price: { $lt: 100 } });
 *   const results = useSearch('products', searchTerm, 5);
 *
 *   return <div>...</div>;
 * }
 */

import { lanceStore } from './lanceql.esm.js';

// Global store cache to share across components
const storeCache = new Map();

/**
 * Get or create a store instance.
 * @param {string} name - Store name
 * @param {Object} options - Store options
 * @returns {Promise<Store>}
 */
async function getStore(name, options = {}) {
    const key = `${name}:${JSON.stringify(options)}`;
    if (!storeCache.has(key)) {
        storeCache.set(key, lanceStore(name, options));
    }
    return storeCache.get(key);
}

/**
 * useStore - React hook for key-value storage.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Key to read/write
 * @param {Object} options - Hook options
 * @returns {Object} { data, set, loading, error }
 *
 * @example
 * function Profile() {
 *   const { data: user, set, loading } = useStore('myapp', 'user');
 *
 *   if (loading) return <div>Loading...</div>;
 *
 *   return (
 *     <div>
 *       <h1>{user?.name}</h1>
 *       <button onClick={() => set({ ...user, name: 'New Name' })}>
 *         Update
 *       </button>
 *     </div>
 *   );
 * }
 */
export function useStore(storeName, key, options = {}) {
    // Check if React is available
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useStore requires React. Import React before using this hook.');
    }

    const { useState, useEffect, useCallback, useRef } = window.React;

    const [data, setData] = useState(options.initialValue);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const storeRef = useRef(null);

    // Load initial value
    useEffect(() => {
        let mounted = true;

        (async () => {
            try {
                const store = await getStore(storeName, options.storeOptions);
                storeRef.current = store;

                const value = await store.get(key);
                if (mounted) {
                    setData(value);
                    setLoading(false);
                }
            } catch (e) {
                if (mounted) {
                    setError(e);
                    setLoading(false);
                }
            }
        })();

        return () => { mounted = false; };
    }, [storeName, key]);

    // Set function
    const set = useCallback(async (value) => {
        try {
            setLoading(true);
            const store = storeRef.current || await getStore(storeName, options.storeOptions);
            await store.set(key, value);
            setData(value);
            setLoading(false);
        } catch (e) {
            setError(e);
            setLoading(false);
        }
    }, [storeName, key]);

    // Delete function
    const remove = useCallback(async () => {
        try {
            const store = storeRef.current || await getStore(storeName, options.storeOptions);
            await store.delete(key);
            setData(undefined);
        } catch (e) {
            setError(e);
        }
    }, [storeName, key]);

    return { data, set, remove, loading, error };
}

/**
 * useQuery - React hook for filtering collections.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {Object} query - Filter query
 * @param {Object} options - Hook options
 * @returns {Object} { data, loading, error, refetch }
 *
 * @example
 * function ProductList() {
 *   const { data: products, loading } = useQuery('myapp', 'products', {
 *     price: { $lt: 100 },
 *     inStock: true
 *   });
 *
 *   if (loading) return <div>Loading...</div>;
 *
 *   return (
 *     <ul>
 *       {products.map(p => <li key={p.id}>{p.name}</li>)}
 *     </ul>
 *   );
 * }
 */
export function useQuery(storeName, key, query = {}, options = {}) {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useQuery requires React. Import React before using this hook.');
    }

    const { useState, useEffect, useCallback } = window.React;

    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetch = useCallback(async () => {
        try {
            setLoading(true);
            const store = await getStore(storeName, options.storeOptions);
            const results = await store.filter(key, query);
            setData(results);
            setLoading(false);
        } catch (e) {
            setError(e);
            setLoading(false);
        }
    }, [storeName, key, JSON.stringify(query)]);

    useEffect(() => {
        fetch();
    }, [fetch]);

    return { data, loading, error, refetch: fetch };
}

/**
 * useSearch - React hook for semantic search.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {string} searchText - Search query
 * @param {number} limit - Max results
 * @param {Object} options - Hook options
 * @returns {Object} { results, loading, error }
 *
 * @example
 * function Search() {
 *   const [query, setQuery] = useState('');
 *   const { results, loading } = useSearch('myapp', 'products', query, 5);
 *
 *   return (
 *     <div>
 *       <input
 *         value={query}
 *         onChange={e => setQuery(e.target.value)}
 *         placeholder="Search products..."
 *       />
 *       <ul>
 *         {results.map(r => (
 *           <li key={r.item.id}>
 *             {r.item.name} (score: {r.score.toFixed(2)})
 *           </li>
 *         ))}
 *       </ul>
 *     </div>
 *   );
 * }
 */
export function useSearch(storeName, key, searchText, limit = 10, options = {}) {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useSearch requires React. Import React before using this hook.');
    }

    const { useState, useEffect, useRef } = window.React;

    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const debounceRef = useRef(null);

    useEffect(() => {
        // Debounce search
        if (debounceRef.current) {
            clearTimeout(debounceRef.current);
        }

        if (!searchText || searchText.trim().length === 0) {
            setResults([]);
            setLoading(false);
            return;
        }

        setLoading(true);

        debounceRef.current = setTimeout(async () => {
            try {
                const store = await getStore(storeName, options.storeOptions);
                const searchResults = await store.search(key, searchText, limit);
                setResults(searchResults);
                setLoading(false);
            } catch (e) {
                setError(e);
                setLoading(false);
            }
        }, options.debounce || 300);

        return () => {
            if (debounceRef.current) {
                clearTimeout(debounceRef.current);
            }
        };
    }, [storeName, key, searchText, limit]);

    return { results, loading, error };
}

/**
 * useStoreKeys - React hook to list all keys in a store.
 *
 * @param {string} storeName - Store name
 * @returns {Object} { keys, loading, error }
 */
export function useStoreKeys(storeName, options = {}) {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useStoreKeys requires React. Import React before using this hook.');
    }

    const { useState, useEffect } = window.React;

    const [keys, setKeys] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        (async () => {
            try {
                const store = await getStore(storeName, options.storeOptions);
                const allKeys = await store.keys();
                setKeys(allKeys);
                setLoading(false);
            } catch (e) {
                setError(e);
                setLoading(false);
            }
        })();
    }, [storeName]);

    return { keys, loading, error };
}

/**
 * StoreProvider - React context provider for store configuration.
 *
 * @example
 * import { StoreProvider, useStore } from '@metal0/lanceql/react';
 *
 * function App() {
 *   return (
 *     <StoreProvider name="myapp">
 *       <Profile />
 *     </StoreProvider>
 *   );
 * }
 *
 * function Profile() {
 *   // Uses 'myapp' store from context
 *   const { data } = useStore('user');
 * }
 */
export function createStoreContext() {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('createStoreContext requires React.');
    }

    const { createContext, useContext } = window.React;
    const StoreContext = createContext(null);

    function StoreProvider({ name, options = {}, children }) {
        return window.React.createElement(
            StoreContext.Provider,
            { value: { name, options } },
            children
        );
    }

    function useStoreContext() {
        return useContext(StoreContext);
    }

    return { StoreProvider, useStoreContext, StoreContext };
}

/**
 * useSemanticSearch - React hook for WebGPU-accelerated semantic search.
 *
 * Automatically enables semantic search and loads the model on first use.
 * Uses GPU-accelerated text encoding for fast similarity search.
 *
 * @param {string} storeName - Store name
 * @param {string} key - Collection key
 * @param {string} searchText - Search query
 * @param {Object} options - Hook options
 * @param {string} options.model - Model name ('minilm', 'clip', or GGUF URL)
 * @param {number} options.limit - Max results (default 10)
 * @param {number} options.debounce - Debounce delay in ms (default 300)
 * @returns {Object} { results, loading, modelLoading, modelInfo, error }
 *
 * @example
 * function Search() {
 *   const [query, setQuery] = useState('');
 *   const { results, loading, modelLoading, modelInfo } = useSemanticSearch(
 *     'myapp', 'products', query, { model: 'minilm', limit: 5 }
 *   );
 *
 *   if (modelLoading) return <div>Loading AI model...</div>;
 *
 *   return (
 *     <div>
 *       <p>Model: {modelInfo?.model} ({modelInfo?.dimensions}D)</p>
 *       <input
 *         value={query}
 *         onChange={e => setQuery(e.target.value)}
 *         placeholder="Search with AI..."
 *       />
 *       {loading ? <div>Searching...</div> : (
 *         <ul>
 *           {results.map(r => (
 *             <li key={r.item.id}>
 *               {r.item.name} (score: {(r.score * 100).toFixed(0)}%)
 *             </li>
 *           ))}
 *         </ul>
 *       )}
 *     </div>
 *   );
 * }
 */
export function useSemanticSearch(storeName, key, searchText, options = {}) {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useSemanticSearch requires React. Import React before using this hook.');
    }

    const { useState, useEffect, useRef, useCallback } = window.React;

    const { model = 'minilm', limit = 10, debounce = 300 } = options;

    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [modelLoading, setModelLoading] = useState(true);
    const [modelInfo, setModelInfo] = useState(null);
    const [error, setError] = useState(null);
    const debounceRef = useRef(null);
    const storeRef = useRef(null);

    // Initialize semantic search on mount
    useEffect(() => {
        let mounted = true;

        (async () => {
            try {
                const store = await getStore(storeName, options.storeOptions);
                storeRef.current = store;

                // Enable semantic search if not already enabled
                if (!store.hasSemanticSearch()) {
                    const info = await store.enableSemanticSearch({
                        model,
                        onProgress: options.onModelProgress
                    });

                    if (mounted && info) {
                        setModelInfo(info);
                    }
                } else {
                    if (mounted) {
                        setModelInfo({
                            model: store._embedder?.model,
                            dimensions: store._embedder?.dimensions
                        });
                    }
                }

                if (mounted) {
                    setModelLoading(false);
                }
            } catch (e) {
                if (mounted) {
                    setError(e);
                    setModelLoading(false);
                }
            }
        })();

        return () => { mounted = false; };
    }, [storeName, model]);

    // Perform search with debouncing
    useEffect(() => {
        if (modelLoading) return;

        if (debounceRef.current) {
            clearTimeout(debounceRef.current);
        }

        if (!searchText || searchText.trim().length === 0) {
            setResults([]);
            setLoading(false);
            return;
        }

        setLoading(true);

        debounceRef.current = setTimeout(async () => {
            try {
                const store = storeRef.current || await getStore(storeName, options.storeOptions);
                const searchResults = await store.search(key, searchText, limit);
                setResults(searchResults);
                setLoading(false);
            } catch (e) {
                setError(e);
                setLoading(false);
            }
        }, debounce);

        return () => {
            if (debounceRef.current) {
                clearTimeout(debounceRef.current);
            }
        };
    }, [searchText, modelLoading, limit, debounce]);

    return { results, loading, modelLoading, modelInfo, error };
}

/**
 * useGPUInfo - React hook to get WebGPU device information.
 *
 * @returns {Object} { info, loading, available }
 *
 * @example
 * function GPUStatus() {
 *   const { info, loading, available } = useGPUInfo();
 *
 *   if (loading) return <div>Checking GPU...</div>;
 *   if (!available) return <div>WebGPU not available</div>;
 *
 *   return (
 *     <div>
 *       <p>GPU: {info.device}</p>
 *       <p>Vendor: {info.vendor}</p>
 *     </div>
 *   );
 * }
 */
export function useGPUInfo() {
    if (typeof window === 'undefined' || !window.React) {
        throw new Error('useGPUInfo requires React. Import React before using this hook.');
    }

    const { useState, useEffect } = window.React;

    const [info, setInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [available, setAvailable] = useState(false);

    useEffect(() => {
        let mounted = true;

        (async () => {
            try {
                // Dynamic import to avoid loading WebGPU module when not needed
                const webgpu = await import('./webgpu/index.js');

                if (!webgpu.isWebGPUAvailable()) {
                    if (mounted) {
                        setAvailable(false);
                        setLoading(false);
                    }
                    return;
                }

                const gpuInfo = await webgpu.getWebGPUInfo();
                if (mounted) {
                    setInfo(gpuInfo);
                    setAvailable(gpuInfo !== null);
                    setLoading(false);
                }
            } catch (e) {
                if (mounted) {
                    setAvailable(false);
                    setLoading(false);
                }
            }
        })();

        return () => { mounted = false; };
    }, []);

    return { info, loading, available };
}
