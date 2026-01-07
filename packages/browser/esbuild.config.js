#!/usr/bin/env node
/**
 * esbuild configuration for LanceQL browser package.
 * Bundles modular source files into distributable bundles.
 */

import * as esbuild from 'esbuild';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, 'dist');

// Ensure dist directory exists
if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
}

const isProduction = process.env.NODE_ENV === 'production';
const isWatch = process.argv.includes('--watch');

// Common build options
const commonOptions = {
    bundle: true,
    sourcemap: true,
    minify: true,
    target: ['es2020'],
    logLevel: 'info',
    loader: {
        '.wgsl': 'text',  // Load WGSL shaders as text
    },
};

async function build() {
    try {
        // Worker bundle (single file for SharedWorker)
        await esbuild.build({
            ...commonOptions,
            entryPoints: ['src/worker/index.js'],
            format: 'esm',
            outfile: 'dist/lanceql-worker.js',
        });
        console.log('✓ Built: dist/lanceql-worker.js');

        // Check if client module exists
        const clientExists = fs.existsSync(path.join(__dirname, 'src/client/index.js'));

        if (clientExists) {
            // Client ESM bundle
            await esbuild.build({
                ...commonOptions,
                entryPoints: ['src/client/index.js'],
                format: 'esm',
                outfile: 'dist/lanceql.esm.js',
            });
            console.log('✓ Built: dist/lanceql.esm.js');

            // Client CJS bundle
            await esbuild.build({
                ...commonOptions,
                entryPoints: ['src/client/index.js'],
                format: 'cjs',
                outfile: 'dist/lanceql.js',
            });
            console.log('✓ Built: dist/lanceql.js');
        } else {
            // Use legacy build for client (copy original files)
            console.log('⚠ Client not yet modularized, using legacy files');
            fs.copyFileSync(
                path.join(__dirname, 'src', 'lanceql.js'),
                path.join(distDir, 'lanceql.esm.js')
            );

            // Create CJS version from ESM
            let content = fs.readFileSync(path.join(__dirname, 'src', 'lanceql.js'), 'utf8');
            content = content
                .replace(/^export\s+const\s+(\w+)\s*=/gm, 'const $1 =')
                .replace(/^export\s+class\s+(\w+)/gm, 'class $1')
                .replace(/^export\s+async\s+function\s+(\w+)/gm, 'async function $1')
                .replace(/^export\s+function\s+(\w+)/gm, 'function $1')
                .replace(/^export\s+default\s+(\w+);?$/gm, '')
                .replace(/^export\s+\{[^}]+\};?$/gm, '');
            fs.writeFileSync(path.join(distDir, 'lanceql.js'), content);
            console.log('✓ Copied: dist/lanceql.esm.js (legacy)');
            console.log('✓ Created: dist/lanceql.js (legacy)');
        }

        // Copy TypeScript definitions
        const dtsFile = path.join(__dirname, 'src', 'lanceql.d.ts');
        if (fs.existsSync(dtsFile)) {
            fs.copyFileSync(dtsFile, path.join(distDir, 'lanceql.d.ts'));
            console.log('✓ Copied: dist/lanceql.d.ts');
        }

        // Copy WebGPU shaders directory
        const webgpuSrcDir = path.join(__dirname, 'src', 'webgpu');
        const webgpuDistDir = path.join(distDir, 'webgpu');
        if (fs.existsSync(webgpuSrcDir)) {
            copyDirRecursive(webgpuSrcDir, webgpuDistDir);
            console.log('✓ Copied: dist/webgpu/');
        }

        console.log('\nBuild complete!');
    } catch (error) {
        console.error('Build failed:', error);
        process.exit(1);
    }
}

function copyDirRecursive(src, dest) {
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDirRecursive(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

if (isWatch) {
    // Watch mode
    const ctx = await esbuild.context({
        ...commonOptions,
        entryPoints: ['src/worker/index.js', 'src/client/index.js'],
        format: 'esm',
        outdir: 'dist',
    });
    await ctx.watch();
    console.log('Watching for changes...');
} else {
    await build();
}
