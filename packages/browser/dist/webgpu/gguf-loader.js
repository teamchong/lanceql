/**
 * GGUF Model Loader
 *
 * Parses GGUF model files and extracts weights for WebGPU inference.
 * Supports any transformer-based text or image encoder.
 *
 * GGUF Format:
 * - Magic: "GGUF" (4 bytes)
 * - Version: u32
 * - Tensor count: u64
 * - Metadata KV count: u64
 * - Metadata key-value pairs
 * - Tensor info (name, dims, type, offset)
 * - Tensor data (aligned)
 */

// GGUF data types
const GGUF_TYPE = {
    UINT8: 0,
    INT8: 1,
    UINT16: 2,
    INT16: 3,
    UINT32: 4,
    INT32: 5,
    FLOAT32: 6,
    BOOL: 7,
    STRING: 8,
    ARRAY: 9,
    UINT64: 10,
    INT64: 11,
    FLOAT64: 12,
};

// GGML quantization types
const GGML_TYPE = {
    F32: 0,
    F16: 1,
    Q4_0: 2,
    Q4_1: 3,
    Q5_0: 6,
    Q5_1: 7,
    Q8_0: 8,
    Q8_1: 9,
    Q2_K: 10,
    Q3_K: 11,
    Q4_K: 12,
    Q5_K: 13,
    Q6_K: 14,
    Q8_K: 15,
    I8: 16,
    I16: 17,
    I32: 18,
    I64: 19,
    F64: 20,
    BF16: 21,
};

// Type sizes in bytes
const TYPE_SIZE = {
    [GGML_TYPE.F32]: 4,
    [GGML_TYPE.F16]: 2,
    [GGML_TYPE.BF16]: 2,
    [GGML_TYPE.Q8_0]: 1 + 16 * 2,  // Block size 32
    [GGML_TYPE.Q4_0]: 2 + 16,       // Block size 32
    [GGML_TYPE.Q4_1]: 2 + 2 + 16,   // Block size 32
};

const BLOCK_SIZE = 32;

/**
 * Parse GGUF file and extract model configuration and weights.
 */
export class GGUFLoader {
    constructor() {
        this.metadata = {};
        this.tensors = new Map();
        this.tensorInfo = [];
        this.dataOffset = 0;
        this.buffer = null;
    }

    /**
     * Load GGUF from URL (streaming for large files).
     * @param {string} url - Model URL
     * @param {Function} onProgress - Progress callback (loaded, total)
     * @returns {Promise<{metadata, tensors}>}
     */
    async loadFromUrl(url, onProgress = null) {
        console.log(`[GGUFLoader] Loading: ${url}`);
        const startTime = performance.now();

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status}`);
        }

        const contentLength = parseInt(response.headers.get('content-length') || '0');
        const reader = response.body.getReader();

        // Read chunks
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            chunks.push(value);
            loaded += value.length;

            if (onProgress) {
                onProgress(loaded, contentLength);
            }
        }

        // Combine chunks
        const totalSize = chunks.reduce((acc, c) => acc + c.length, 0);
        this.buffer = new Uint8Array(totalSize);
        let offset = 0;
        for (const chunk of chunks) {
            this.buffer.set(chunk, offset);
            offset += chunk.length;
        }

        const loadTime = performance.now() - startTime;
        console.log(`[GGUFLoader] Downloaded ${(totalSize / 1e6).toFixed(1)}MB in ${(loadTime / 1000).toFixed(1)}s`);

        return this.parse();
    }

    /**
     * Load GGUF from ArrayBuffer.
     * @param {ArrayBuffer} buffer - Model data
     * @returns {{metadata, tensors}}
     */
    loadFromBuffer(buffer) {
        this.buffer = new Uint8Array(buffer);
        return this.parse();
    }

    /**
     * Parse the GGUF file.
     */
    parse() {
        const view = new DataView(this.buffer.buffer);
        let offset = 0;

        // Magic
        const magic = String.fromCharCode(...this.buffer.slice(0, 4));
        if (magic !== 'GGUF') {
            throw new Error(`Invalid GGUF magic: ${magic}`);
        }
        offset += 4;

        // Version
        const version = view.getUint32(offset, true);
        offset += 4;
        console.log(`[GGUFLoader] GGUF version: ${version}`);

        // Tensor count
        const tensorCount = Number(view.getBigUint64(offset, true));
        offset += 8;

        // Metadata count
        const metadataCount = Number(view.getBigUint64(offset, true));
        offset += 8;

        console.log(`[GGUFLoader] Tensors: ${tensorCount}, Metadata: ${metadataCount}`);

        // Parse metadata
        for (let i = 0; i < metadataCount; i++) {
            const { key, value, newOffset } = this._parseKV(view, offset);
            this.metadata[key] = value;
            offset = newOffset;
        }

        // Parse tensor info
        for (let i = 0; i < tensorCount; i++) {
            const { info, newOffset } = this._parseTensorInfo(view, offset);
            this.tensorInfo.push(info);
            offset = newOffset;
        }

        // Align to 32 bytes for tensor data
        this.dataOffset = Math.ceil(offset / 32) * 32;

        console.log(`[GGUFLoader] Data offset: ${this.dataOffset}`);
        console.log(`[GGUFLoader] Model: ${this.metadata['general.name'] || 'unknown'}`);

        return {
            metadata: this.metadata,
            tensors: this.tensorInfo,
        };
    }

    /**
     * Extract model architecture config.
     */
    getModelConfig() {
        const arch = this.metadata['general.architecture'] || 'bert';

        // Common config keys
        const config = {
            architecture: arch,
            name: this.metadata['general.name'] || 'unknown',
            vocabSize: this.metadata[`${arch}.vocab_size`] || 30522,
            hiddenSize: this.metadata[`${arch}.embedding_length`] || 384,
            numLayers: this.metadata[`${arch}.block_count`] || 6,
            numHeads: this.metadata[`${arch}.attention.head_count`] || 12,
            intermediateSize: this.metadata[`${arch}.feed_forward_length`] || 1536,
            maxPositions: this.metadata[`${arch}.context_length`] || 512,
            layerNormEps: this.metadata[`${arch}.attention.layer_norm_epsilon`] || 1e-12,
        };

        // CLIP-specific
        if (arch === 'clip' || this.metadata['clip.vision.embedding_length']) {
            config.visionHiddenSize = this.metadata['clip.vision.embedding_length'] || 768;
            config.visionNumLayers = this.metadata['clip.vision.block_count'] || 12;
            config.visionNumHeads = this.metadata['clip.vision.attention.head_count'] || 12;
            config.imageSize = this.metadata['clip.vision.image_size'] || 224;
            config.patchSize = this.metadata['clip.vision.patch_size'] || 32;
        }

        return config;
    }

    /**
     * Get tensor by name.
     * @param {string} name - Tensor name
     * @returns {Float32Array|null}
     */
    getTensor(name) {
        const info = this.tensorInfo.find(t => t.name === name);
        if (!info) {
            console.warn(`[GGUFLoader] Tensor not found: ${name}`);
            return null;
        }

        return this._extractTensor(info);
    }

    /**
     * Get all weight tensors as Float32Arrays.
     * Dequantizes quantized weights to F32.
     * @returns {Object} name → Float32Array
     */
    getAllWeights() {
        const weights = {};

        for (const info of this.tensorInfo) {
            weights[info.name] = this._extractTensor(info);
        }

        return weights;
    }

    /**
     * Get weight tensor names matching a pattern.
     * @param {RegExp} pattern - Name pattern
     * @returns {string[]}
     */
    getTensorNames(pattern = null) {
        if (!pattern) {
            return this.tensorInfo.map(t => t.name);
        }
        return this.tensorInfo.filter(t => pattern.test(t.name)).map(t => t.name);
    }

    // =========================================================================
    // Internal parsing methods
    // =========================================================================

    _parseKV(view, offset) {
        // Key (string)
        const keyLen = Number(view.getBigUint64(offset, true));
        offset += 8;
        const key = new TextDecoder().decode(this.buffer.slice(offset, offset + keyLen));
        offset += keyLen;

        // Value type
        const valueType = view.getUint32(offset, true);
        offset += 4;

        // Value
        const { value, newOffset } = this._parseValue(view, offset, valueType);

        return { key, value, newOffset };
    }

    _parseValue(view, offset, type) {
        switch (type) {
            case GGUF_TYPE.UINT8:
                return { value: view.getUint8(offset), newOffset: offset + 1 };
            case GGUF_TYPE.INT8:
                return { value: view.getInt8(offset), newOffset: offset + 1 };
            case GGUF_TYPE.UINT16:
                return { value: view.getUint16(offset, true), newOffset: offset + 2 };
            case GGUF_TYPE.INT16:
                return { value: view.getInt16(offset, true), newOffset: offset + 2 };
            case GGUF_TYPE.UINT32:
                return { value: view.getUint32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.INT32:
                return { value: view.getInt32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.FLOAT32:
                return { value: view.getFloat32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.UINT64:
                return { value: Number(view.getBigUint64(offset, true)), newOffset: offset + 8 };
            case GGUF_TYPE.INT64:
                return { value: Number(view.getBigInt64(offset, true)), newOffset: offset + 8 };
            case GGUF_TYPE.FLOAT64:
                return { value: view.getFloat64(offset, true), newOffset: offset + 8 };
            case GGUF_TYPE.BOOL:
                return { value: view.getUint8(offset) !== 0, newOffset: offset + 1 };
            case GGUF_TYPE.STRING: {
                const len = Number(view.getBigUint64(offset, true));
                offset += 8;
                const str = new TextDecoder().decode(this.buffer.slice(offset, offset + len));
                return { value: str, newOffset: offset + len };
            }
            case GGUF_TYPE.ARRAY: {
                const elemType = view.getUint32(offset, true);
                offset += 4;
                const count = Number(view.getBigUint64(offset, true));
                offset += 8;
                const arr = [];
                for (let i = 0; i < count; i++) {
                    const { value, newOffset } = this._parseValue(view, offset, elemType);
                    arr.push(value);
                    offset = newOffset;
                }
                return { value: arr, newOffset: offset };
            }
            default:
                throw new Error(`Unknown GGUF type: ${type}`);
        }
    }

    _parseTensorInfo(view, offset) {
        // Name
        const nameLen = Number(view.getBigUint64(offset, true));
        offset += 8;
        const name = new TextDecoder().decode(this.buffer.slice(offset, offset + nameLen));
        offset += nameLen;

        // Dimensions
        const nDims = view.getUint32(offset, true);
        offset += 4;
        const dims = [];
        for (let i = 0; i < nDims; i++) {
            dims.push(Number(view.getBigUint64(offset, true)));
            offset += 8;
        }

        // Type
        const type = view.getUint32(offset, true);
        offset += 4;

        // Offset in data section
        const dataOffset = Number(view.getBigUint64(offset, true));
        offset += 8;

        // Calculate size
        const numElements = dims.reduce((a, b) => a * b, 1);

        return {
            info: { name, dims, type, dataOffset, numElements },
            newOffset: offset,
        };
    }

    _extractTensor(info) {
        const start = this.dataOffset + info.dataOffset;

        switch (info.type) {
            case GGML_TYPE.F32: {
                const data = new Float32Array(
                    this.buffer.buffer,
                    this.buffer.byteOffset + start,
                    info.numElements
                );
                return new Float32Array(data);  // Copy to avoid detached buffer issues
            }

            case GGML_TYPE.F16: {
                return this._dequantizeF16(start, info.numElements);
            }

            case GGML_TYPE.BF16: {
                return this._dequantizeBF16(start, info.numElements);
            }

            case GGML_TYPE.Q8_0: {
                return this._dequantizeQ8_0(start, info.numElements);
            }

            case GGML_TYPE.Q4_0: {
                return this._dequantizeQ4_0(start, info.numElements);
            }

            default:
                console.warn(`[GGUFLoader] Unsupported tensor type: ${info.type} for ${info.name}`);
                return new Float32Array(info.numElements);
        }
    }

    _dequantizeF16(start, numElements) {
        const result = new Float32Array(numElements);
        const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + start);

        for (let i = 0; i < numElements; i++) {
            const h = view.getUint16(i * 2, true);
            result[i] = this._fp16ToFp32(h);
        }

        return result;
    }

    _dequantizeBF16(start, numElements) {
        const result = new Float32Array(numElements);
        const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + start);

        for (let i = 0; i < numElements; i++) {
            const b = view.getUint16(i * 2, true);
            // BF16 is just the upper 16 bits of F32
            const f32bits = b << 16;
            const arr = new Float32Array(1);
            new Uint32Array(arr.buffer)[0] = f32bits;
            result[i] = arr[0];
        }

        return result;
    }

    _dequantizeQ8_0(start, numElements) {
        const result = new Float32Array(numElements);
        const numBlocks = Math.ceil(numElements / BLOCK_SIZE);
        const blockSize = 2 + BLOCK_SIZE;  // scale (f16) + 32 int8 values

        for (let block = 0; block < numBlocks; block++) {
            const blockStart = start + block * blockSize;
            const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + blockStart);

            // Scale (f16)
            const scale = this._fp16ToFp32(view.getUint16(0, true));

            // Dequantize values
            for (let i = 0; i < BLOCK_SIZE; i++) {
                const idx = block * BLOCK_SIZE + i;
                if (idx >= numElements) break;

                const q = view.getInt8(2 + i);
                result[idx] = q * scale;
            }
        }

        return result;
    }

    _dequantizeQ4_0(start, numElements) {
        const result = new Float32Array(numElements);
        const numBlocks = Math.ceil(numElements / BLOCK_SIZE);
        const blockSize = 2 + BLOCK_SIZE / 2;  // scale (f16) + 16 bytes (32 nibbles)

        for (let block = 0; block < numBlocks; block++) {
            const blockStart = start + block * blockSize;
            const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + blockStart);

            // Scale (f16)
            const scale = this._fp16ToFp32(view.getUint16(0, true));

            // Dequantize values (4-bit quantized)
            for (let i = 0; i < BLOCK_SIZE; i++) {
                const idx = block * BLOCK_SIZE + i;
                if (idx >= numElements) break;

                const byteIdx = Math.floor(i / 2);
                const byte = view.getUint8(2 + byteIdx);

                // Each byte contains 2 4-bit values
                let q;
                if (i % 2 === 0) {
                    q = (byte & 0x0F) - 8;
                } else {
                    q = ((byte >> 4) & 0x0F) - 8;
                }

                result[idx] = q * scale;
            }
        }

        return result;
    }

    _fp16ToFp32(h) {
        const sign = (h >> 15) & 1;
        const exp = (h >> 10) & 0x1F;
        const mant = h & 0x3FF;

        if (exp === 0) {
            if (mant === 0) return sign ? -0 : 0;
            // Subnormal
            const val = mant / 1024 * Math.pow(2, -14);
            return sign ? -val : val;
        }

        if (exp === 31) {
            if (mant === 0) return sign ? -Infinity : Infinity;
            return NaN;
        }

        const val = (1 + mant / 1024) * Math.pow(2, exp - 15);
        return sign ? -val : val;
    }
}

/**
 * Known model URLs for common encoders.
 */
export const MODEL_REGISTRY = {
    // Text encoders
    'minilm': 'https://data.metal0.dev/models/minilm-l6-v2.gguf',
    'minilm-l6': 'https://data.metal0.dev/models/minilm-l6-v2.gguf',
    'minilm-l12': 'https://data.metal0.dev/models/minilm-l12-v2.gguf',

    // CLIP text encoders
    'clip': 'https://data.metal0.dev/models/clip-vit-b32-openai.gguf',
    'clip-openai': 'https://data.metal0.dev/models/clip-vit-b32-openai.gguf',
    'clip-laion': 'https://data.metal0.dev/models/clip-vit-b32-laion.gguf',

    // Other text encoders
    'bge-small': 'https://data.metal0.dev/models/bge-small-en-v1.5.gguf',
    'bge-base': 'https://data.metal0.dev/models/bge-base-en-v1.5.gguf',
    'e5-small': 'https://data.metal0.dev/models/e5-small-v2.gguf',
};

/**
 * Resolve model name to URL.
 * @param {string} nameOrUrl - Model name or URL
 * @returns {string} URL
 */
export function resolveModelUrl(nameOrUrl) {
    if (nameOrUrl.startsWith('http://') || nameOrUrl.startsWith('https://')) {
        return nameOrUrl;
    }

    const url = MODEL_REGISTRY[nameOrUrl.toLowerCase()];
    if (!url) {
        throw new Error(`Unknown model: ${nameOrUrl}. Use a URL or one of: ${Object.keys(MODEL_REGISTRY).join(', ')}`);
    }

    return url;
}
