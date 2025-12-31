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

console.log('Build complete!');
