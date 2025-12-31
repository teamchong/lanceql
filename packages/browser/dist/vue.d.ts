/**
 * LanceQL Vue Bindings
 */
import type { Ref } from 'vue';

export interface UseStoreResult<T> {
    data: Ref<T | undefined>;
    set: (value: T) => Promise<void>;
    remove: () => Promise<void>;
    loading: Ref<boolean>;
    error: Ref<Error | null>;
}

export interface UseQueryResult<T> {
    data: Ref<T[]>;
    loading: Ref<boolean>;
    error: Ref<Error | null>;
    refetch: () => Promise<void>;
}

export interface UseSearchResult<T> {
    results: Ref<Array<{ item: T; score: number }>>;
    query: Ref<string>;
    search: (text: string) => Promise<void>;
    loading: Ref<boolean>;
    error: Ref<Error | null>;
    clear: () => void;
}

export interface UseCollectionResult<T> {
    items: Ref<T[]>;
    add: (item: T) => Promise<void>;
    remove: (predicate: ((item: T) => boolean) | string | number | { id: any }) => Promise<void>;
    update: (id: any, updates: Partial<T>) => Promise<void>;
    clear: () => Promise<void>;
    reload: () => Promise<void>;
    loading: Ref<boolean>;
    error: Ref<Error | null>;
}

export interface StoreOptions {
    initialValue?: any;
    storeOptions?: {
        session?: boolean;
    };
}

export function useStore<T = any>(
    storeName: string,
    key: string,
    options?: StoreOptions
): UseStoreResult<T>;

export function useQuery<T = any>(
    storeName: string,
    key: string,
    query?: Record<string, any> | Ref<Record<string, any>>,
    options?: StoreOptions
): UseQueryResult<T>;

export function useSearch<T = any>(
    storeName: string,
    key: string,
    options?: { limit?: number; debounce?: number; storeOptions?: { session?: boolean } }
): UseSearchResult<T>;

export function useCollection<T = any>(
    storeName: string,
    key: string,
    options?: StoreOptions
): UseCollectionResult<T>;

export function useStoreKeys(
    storeName: string,
    options?: StoreOptions
): { keys: Ref<string[]>; loading: Ref<boolean>; error: Ref<Error | null> };
