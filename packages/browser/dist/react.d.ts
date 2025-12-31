/**
 * LanceQL React Bindings
 */
import type { Store } from './lanceql';

export interface UseStoreResult<T> {
    data: T | undefined;
    set: (value: T) => Promise<void>;
    remove: () => Promise<void>;
    loading: boolean;
    error: Error | null;
}

export interface UseQueryResult<T> {
    data: T[];
    loading: boolean;
    error: Error | null;
    refetch: () => Promise<void>;
}

export interface UseSearchResult<T> {
    results: Array<{ item: T; score: number }>;
    loading: boolean;
    error: Error | null;
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
    query?: Record<string, any>,
    options?: StoreOptions
): UseQueryResult<T>;

export function useSearch<T = any>(
    storeName: string,
    key: string,
    searchText: string,
    limit?: number,
    options?: { debounce?: number; storeOptions?: { session?: boolean } }
): UseSearchResult<T>;

export function useStoreKeys(
    storeName: string,
    options?: StoreOptions
): { keys: string[]; loading: boolean; error: Error | null };

export function createStoreContext(): {
    StoreProvider: React.FC<{ name: string; options?: any; children: React.ReactNode }>;
    useStoreContext: () => { name: string; options: any } | null;
    StoreContext: React.Context<{ name: string; options: any } | null>;
};
