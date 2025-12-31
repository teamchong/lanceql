/**
 * LanceQL Svelte Bindings
 */
import type { Readable, Writable } from 'svelte/store';

export interface LanceStoreValue<T> {
    value: T | undefined;
    loading: boolean;
    error: Error | null;
}

export interface QueryStoreValue<T> {
    items: T[];
    loading: boolean;
    error: Error | null;
    query: Record<string, any>;
}

export interface SearchStoreValue<T> {
    query: string;
    results: Array<{ item: T; score: number }>;
    loading: boolean;
    error: Error | null;
}

export interface CollectionStoreValue<T> {
    items: T[];
    loading: boolean;
    error: Error | null;
}

export function createLanceStore<T = any>(
    storeName: string,
    key: string,
    initialValue?: T
): Writable<LanceStoreValue<T>>;

export function createQueryStore<T = any>(
    storeName: string,
    key: string,
    initialQuery?: Record<string, any>
): Readable<QueryStoreValue<T>> & {
    refetch: () => Promise<void>;
    setQuery: (query: Record<string, any>) => Promise<void>;
};

export function createSearchStore<T = any>(
    storeName: string,
    key: string,
    options?: { limit?: number; debounce?: number }
): Readable<SearchStoreValue<T>> & {
    search: (text: string) => Promise<void>;
    clear: () => void;
};

export function createCollectionStore<T = any>(
    storeName: string,
    key: string
): Readable<CollectionStoreValue<T>> & {
    add: (item: T) => Promise<void>;
    remove: (predicate: ((item: T) => boolean) | string | number | { id: any }) => Promise<void>;
    update: (id: any, updates: Partial<T>) => Promise<void>;
    clear: () => Promise<void>;
    reload: () => Promise<void>;
};
