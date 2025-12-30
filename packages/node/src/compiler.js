/**
 * Runtime @logic_table compiler using metal0.
 *
 * Compiles Python @logic_table classes to native shared libraries (.so/.dylib)
 * at runtime, enabling high-performance vectorized operations without
 * ahead-of-time compilation.
 *
 * @example
 * const { compileLogicTable } = require('@metal0/lanceql-node');
 *
 * const source = `
 * from logic_table import logic_table
 *
 * @logic_table
 * class VectorOps:
 *     def dot_product(self, a: list, b: list) -> float:
 *         result = 0.0
 *         for i in range(len(a)):
 *             result = result + a[i] * b[i]
 *         return result
 * `;
 *
 * const ops = await compileLogicTable(source);
 * const result = ops.VectorOps.dot_product([1.0, 2.0], [3.0, 4.0]);  // Returns 11.0
 */

const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

/**
 * Error thrown when @logic_table compilation fails.
 */
class CompilerError extends Error {
    constructor(message) {
        super(message);
        this.name = 'CompilerError';
    }
}

/**
 * Get the platform-specific shared library extension.
 */
function getLibExtension() {
    switch (process.platform) {
        case 'darwin': return '.dylib';
        case 'win32': return '.dll';
        default: return '.so';
    }
}

/**
 * Get the default cache directory.
 */
function getDefaultCacheDir() {
    if (process.platform === 'darwin') {
        return path.join(os.homedir(), 'Library', 'Caches', 'metal0', 'logic_table');
    } else if (process.platform === 'win32') {
        return path.join(process.env.LOCALAPPDATA || os.homedir(), 'metal0', 'cache', 'logic_table');
    } else {
        return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'metal0', 'logic_table');
    }
}

/**
 * Find the metal0 binary.
 */
function findMetal0(providedPath) {
    // 1. Use provided path
    if (providedPath && fs.existsSync(providedPath)) {
        return path.resolve(providedPath);
    }

    // 2. Check environment variable
    const envPath = process.env.METAL0_PATH;
    if (envPath && fs.existsSync(envPath)) {
        return path.resolve(envPath);
    }

    // 3. Check bundled binary relative to this package
    // IMPORTANT: Use path.resolve() to normalize paths with .. components
    const packageDir = path.resolve(__dirname);
    const bundledPaths = [
        path.join(packageDir, 'bin', process.platform === 'win32' ? 'metal0.exe' : 'metal0'),
        path.join(packageDir, '..', '..', '..', 'deps', 'metal0', 'zig-out', 'bin', 'metal0'),
    ];

    for (const p of bundledPaths) {
        const resolved = path.resolve(p);
        if (fs.existsSync(resolved)) {
            return resolved;
        }
    }

    // 4. Check system PATH
    const pathDirs = (process.env.PATH || '').split(path.delimiter);
    for (const dir of pathDirs) {
        const metal0Path = path.join(dir, process.platform === 'win32' ? 'metal0.exe' : 'metal0');
        if (fs.existsSync(metal0Path)) {
            return path.resolve(metal0Path);
        }
    }

    throw new CompilerError(
        'metal0 binary not found. Install via:\n' +
        '  1. Set METAL0_PATH environment variable, or\n' +
        '  2. Build from source: cd deps/metal0 && zig build, or\n' +
        '  3. Add metal0 to PATH'
    );
}

/**
 * Compute content hash for caching.
 */
function computeHash(source) {
    return crypto.createHash('sha256').update(source, 'utf8').digest('hex').substring(0, 16);
}

/**
 * LogicTableCompiler - Compiles Python @logic_table classes to native shared libraries.
 */
class LogicTableCompiler {
    constructor(options = {}) {
        this.metal0Path = findMetal0(options.metal0Path);
        this.cacheDir = options.cacheDir || getDefaultCacheDir();

        // Ensure cache directory exists
        fs.mkdirSync(this.cacheDir, { recursive: true });
    }

    /**
     * Compile @logic_table source code to native module.
     *
     * @param {string} source - Python source code containing @logic_table decorated class(es)
     * @param {Object} options - Compilation options
     * @param {boolean} options.force - If true, recompile even if cached version exists
     * @returns {Promise<CompiledLogicTable>} - Compiled module with callable methods
     */
    async compile(source, options = {}) {
        const contentHash = computeHash(source);
        const libExt = getLibExtension();
        const cachedLib = path.join(this.cacheDir, `logic_table_${contentHash}${libExt}`);

        // Check cache
        if (!options.force && fs.existsSync(cachedLib)) {
            const stats = fs.statSync(cachedLib);
            if (stats.size > 0) {
                return new CompiledLogicTable(cachedLib);
            }
        }

        // Write source to temp file
        const tempSource = path.join(os.tmpdir(), `logic_table_${contentHash}.py`);
        fs.writeFileSync(tempSource, source, 'utf8');

        try {
            // Determine metal0 working directory
            const metal0Bin = path.resolve(this.metal0Path);
            let metal0Dir = path.dirname(path.dirname(path.dirname(metal0Bin)));  // zig-out/bin/metal0 -> metal0

            // Verify this is the right directory
            if (!fs.existsSync(path.join(metal0Dir, 'build.zig'))) {
                metal0Dir = path.dirname(path.dirname(metal0Bin));
                if (!fs.existsSync(path.join(metal0Dir, 'build.zig'))) {
                    metal0Dir = undefined;
                }
            }

            // Compile using metal0
            const result = await new Promise((resolve, reject) => {
                const args = ['build', '--emit-logic-table-shared', tempSource, '-o', cachedLib];
                const child = spawn(this.metal0Path, args, {
                    cwd: metal0Dir,
                    timeout: 60000,  // 60 second timeout
                });

                let stdout = '';
                let stderr = '';

                child.stdout.on('data', (data) => { stdout += data.toString(); });
                child.stderr.on('data', (data) => { stderr += data.toString(); });

                child.on('close', (code) => {
                    resolve({ code, stdout, stderr });
                });

                child.on('error', (err) => {
                    reject(new CompilerError(`Failed to run metal0: ${err.message}`));
                });
            });

            if (result.code !== 0) {
                const errorMsg = result.stderr || result.stdout || 'Unknown compilation error';
                throw new CompilerError(`Compilation failed:\n${errorMsg}`);
            }

            if (!fs.existsSync(cachedLib)) {
                throw new CompilerError(`Compilation succeeded but output not found: ${cachedLib}`);
            }

            const stats = fs.statSync(cachedLib);
            if (stats.size === 0) {
                const errorMsg = result.stderr || result.stdout;
                throw new CompilerError(`Compilation produced empty library:\n${errorMsg}`);
            }

            return new CompiledLogicTable(cachedLib);

        } finally {
            // Clean up temp file
            try {
                fs.unlinkSync(tempSource);
            } catch (e) {
                // Ignore cleanup errors
            }
        }
    }

    /**
     * Compile @logic_table from a Python file.
     *
     * @param {string} filePath - Path to Python source file
     * @param {Object} options - Compilation options
     * @returns {Promise<CompiledLogicTable>} - Compiled module with callable methods
     */
    async compileFile(filePath, options = {}) {
        if (!fs.existsSync(filePath)) {
            throw new CompilerError(`Source file not found: ${filePath}`);
        }
        const source = fs.readFileSync(filePath, 'utf8');
        return this.compile(source, options);
    }
}

/**
 * CompiledLogicTable - Wrapper for a compiled @logic_table shared library.
 *
 * Provides JavaScript access to native @logic_table functions via N-API/FFI.
 */
class CompiledLogicTable {
    constructor(libPath) {
        if (!fs.existsSync(libPath)) {
            throw new Error(`Library not found: ${libPath}`);
        }

        this._libPath = libPath;
        this._methods = {};
        this._classes = {};

        // Load the library and discover methods
        this._loadLibrary();
    }

    _loadLibrary() {
        // Use the native DynamicLogicTable from binding.node
        try {
            // Try multiple possible locations for the binding
            let binding = null;
            const bindingPaths = [
                '../build/Release/lanceql.node',
                '../build/Debug/lanceql.node',
            ];

            for (const bindingPath of bindingPaths) {
                try {
                    binding = require(bindingPath);
                    break;
                } catch (e) {
                    // Try next path
                }
            }

            if (!binding) {
                throw new Error('Native binding not found. Run: npm run build');
            }

            if (!binding.DynamicLogicTable) {
                throw new Error('DynamicLogicTable not exported from native binding');
            }

            this._native = new binding.DynamicLogicTable(this._libPath);
            this._discoverMethods();
        } catch (e) {
            throw new Error(
                `Failed to load logic_table library: ${e.message}\n` +
                `Library path: ${this._libPath}\n` +
                'Ensure the native binding is built: cd packages/node && npm run build'
            );
        }
    }

    _discoverMethods() {
        if (!this._native) return;

        // Get exported symbols from native binding
        const symbols = this._native.getSymbols();
        for (const symbol of symbols) {
            // Parse ClassName_methodName pattern (split at first underscore only)
            const underscoreIdx = symbol.indexOf('_');
            if (underscoreIdx === -1) continue;

            const className = symbol.substring(0, underscoreIdx);
            const methodName = symbol.substring(underscoreIdx + 1);

            if (!className || !methodName) continue;

            // Filter out internal symbols
            if (['libdeflate', 'std', 'runtime', 'c'].includes(className)) {
                continue;
            }

            // Create method wrapper that calls native function
            const wrapper = (...args) => this._native.call(symbol, args);

            // Store under full name
            this._methods[symbol] = wrapper;

            // Create class proxy if needed
            if (!this._classes[className]) {
                this._classes[className] = {};
            }
            this._classes[className][methodName] = wrapper;
        }

        // Expose classes as properties
        for (const [className, methods] of Object.entries(this._classes)) {
            Object.defineProperty(this, className, {
                value: methods,
                enumerable: true,
            });
        }
    }

    get methods() {
        return { ...this._methods };
    }

    get classes() {
        return { ...this._classes };
    }
}

// Default compiler instance
let _defaultCompiler = null;

function getCompiler() {
    if (!_defaultCompiler) {
        _defaultCompiler = new LogicTableCompiler();
    }
    return _defaultCompiler;
}

/**
 * Compile @logic_table source code to native module.
 *
 * @param {string} source - Python source code containing @logic_table decorated class(es)
 * @param {Object} options - Compilation options
 * @returns {Promise<CompiledLogicTable>} - Compiled module with callable methods
 */
async function compileLogicTable(source, options = {}) {
    return getCompiler().compile(source, options);
}

/**
 * Compile @logic_table from a Python file.
 *
 * @param {string} filePath - Path to Python source file
 * @param {Object} options - Compilation options
 * @returns {Promise<CompiledLogicTable>} - Compiled module with callable methods
 */
async function compileLogicTableFile(filePath, options = {}) {
    return getCompiler().compileFile(filePath, options);
}

module.exports = {
    LogicTableCompiler,
    CompiledLogicTable,
    CompilerError,
    compileLogicTable,
    compileLogicTableFile,
};
