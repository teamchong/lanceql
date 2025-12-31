#!/usr/bin/env node
/**
 * Build script for lanceql npm package.
 * Bundles the JavaScript module and copies necessary files.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const srcDir = path.join(__dirname, '..', 'src');
const distDir = path.join(__dirname, '..', 'dist');

// Ensure dist directory exists
if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
}

// Read the source file
const srcFile = path.join(srcDir, 'lanceql.js');
let content = fs.readFileSync(srcFile, 'utf8');

// Create ESM version (already uses export)
const esmContent = content;
fs.writeFileSync(path.join(distDir, 'lanceql.esm.js'), esmContent);
console.log('Created: dist/lanceql.esm.js');

// Find all named exports
const namedExports = [];
const constMatches = content.matchAll(/^export\s+const\s+(\w+)\s*=/gm);
for (const match of constMatches) namedExports.push(match[1]);

const classMatches = content.matchAll(/^export\s+class\s+(\w+)/gm);
for (const match of classMatches) namedExports.push(match[1]);

const funcMatches = content.matchAll(/^export\s+(?:async\s+)?function\s+(\w+)/gm);
for (const match of funcMatches) namedExports.push(match[1]);

console.log(`Found ${namedExports.length} named exports: ${namedExports.join(', ')}`);

// Create CJS version - remove export keywords
let cjsContent = content
    .replace(/^export\s+const\s+(\w+)\s*=/gm, 'const $1 =')
    .replace(/^export\s+class\s+(\w+)/gm, 'class $1')
    .replace(/^export\s+async\s+function\s+(\w+)/gm, 'async function $1')
    .replace(/^export\s+function\s+(\w+)/gm, 'function $1')
    .replace(/^export\s+default\s+(\w+);?$/gm, '// default export: $1')
    .replace(/^export\s+\{[^}]+\};?$/gm, '');  // Remove named export blocks

// Add module.exports at the end
cjsContent += '\n\n// CommonJS exports\n';
cjsContent += 'module.exports = {\n';
for (const name of namedExports) {
    cjsContent += `    ${name},\n`;
}
cjsContent += `    default: LanceQL\n`;
cjsContent += '};\n';

fs.writeFileSync(path.join(distDir, 'lanceql.js'), cjsContent);
console.log('Created: dist/lanceql.js');

// Copy TypeScript definitions
const dtsFile = path.join(srcDir, 'lanceql.d.ts');
if (fs.existsSync(dtsFile)) {
    fs.copyFileSync(dtsFile, path.join(distDir, 'lanceql.d.ts'));
    console.log('Created: dist/lanceql.d.ts');
}

// Copy LanceQL worker
const workerSrc = path.join(srcDir, 'lanceql-worker.js');
if (fs.existsSync(workerSrc)) {
    fs.copyFileSync(workerSrc, path.join(distDir, 'lanceql-worker.js'));
    console.log('Created: dist/lanceql-worker.js');
}

// Copy WebGPU module directory
const webgpuSrcDir = path.join(srcDir, 'webgpu');
const webgpuDistDir = path.join(distDir, 'webgpu');
if (fs.existsSync(webgpuSrcDir)) {
    // Create webgpu directory in dist
    if (!fs.existsSync(webgpuDistDir)) {
        fs.mkdirSync(webgpuDistDir, { recursive: true });
    }

    // Copy shaders directory
    const shadersSrcDir = path.join(webgpuSrcDir, 'shaders');
    const shadersDistDir = path.join(webgpuDistDir, 'shaders');
    if (fs.existsSync(shadersSrcDir)) {
        if (!fs.existsSync(shadersDistDir)) {
            fs.mkdirSync(shadersDistDir, { recursive: true });
        }
        const shaderFiles = fs.readdirSync(shadersSrcDir);
        for (const file of shaderFiles) {
            fs.copyFileSync(path.join(shadersSrcDir, file), path.join(shadersDistDir, file));
        }
        console.log(`Copied ${shaderFiles.length} shader files to dist/webgpu/shaders/`);
    }

    // Copy JS files
    const jsFiles = fs.readdirSync(webgpuSrcDir).filter(f => f.endsWith('.js'));
    for (const file of jsFiles) {
        fs.copyFileSync(path.join(webgpuSrcDir, file), path.join(webgpuDistDir, file));
    }
    console.log(`Copied ${jsFiles.length} JS files to dist/webgpu/`);
}

// Build framework bindings
const frameworks = ['react', 'svelte', 'vue'];
for (const framework of frameworks) {
    const frameworkSrc = path.join(srcDir, `${framework}.js`);
    if (fs.existsSync(frameworkSrc)) {
        let frameworkContent = fs.readFileSync(frameworkSrc, 'utf8');

        // Update import path for bundled version
        frameworkContent = frameworkContent.replace(
            /from\s+['"]\.\/lanceql\.js['"]/g,
            "from './lanceql.esm.js'"
        );

        fs.writeFileSync(path.join(distDir, `${framework}.js`), frameworkContent);
        console.log(`Created: dist/${framework}.js`);

        // Generate TypeScript definition stub
        const dtsContent = generateFrameworkDts(framework);
        fs.writeFileSync(path.join(distDir, `${framework}.d.ts`), dtsContent);
        console.log(`Created: dist/${framework}.d.ts`);
    }
}

/**
 * Generate TypeScript definitions for framework bindings.
 */
function generateFrameworkDts(framework) {
    switch (framework) {
        case 'react':
            return `/**
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
`;

        case 'svelte':
            return `/**
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
`;

        case 'vue':
            return `/**
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
`;

        default:
            return `// ${framework} bindings\n`;
    }
}

console.log('Build complete!');
