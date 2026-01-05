// src/worker/encryption.js
var encryptionKeys = /* @__PURE__ */ new Map();
async function importEncryptionKey(keyBytes) {
  return crypto.subtle.importKey(
    "raw",
    new Uint8Array(keyBytes),
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}
async function encryptData(data, cryptoKey) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoder = new TextEncoder();
  const plaintext = encoder.encode(JSON.stringify(data));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    cryptoKey,
    plaintext
  );
  const result = new Uint8Array(12 + ciphertext.byteLength);
  result.set(iv, 0);
  result.set(new Uint8Array(ciphertext), 12);
  return result;
}
async function decryptData(encrypted, cryptoKey) {
  const iv = encrypted.slice(0, 12);
  const ciphertext = encrypted.slice(12);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    cryptoKey,
    ciphertext
  );
  const decoder = new TextDecoder();
  return JSON.parse(decoder.decode(plaintext));
}

// src/worker/worker-store.js
var gpuTransformer = null;
var gpuTransformerPromise = null;
var embeddingCache = /* @__PURE__ */ new Map();
function setGPUTransformer(transformer) {
  gpuTransformer = transformer;
}
function setGPUTransformerPromise(promise) {
  gpuTransformerPromise = promise;
}
var WorkerStore = class {
  constructor(name, options = {}) {
    this.name = name;
    this.options = options;
    this._root = null;
    this._ready = false;
    this._embedder = null;
    this._encryptionKeyId = null;
  }
  async open(encryptionConfig = null) {
    if (this._ready) return this;
    if (encryptionConfig) {
      const { keyId, keyBytes } = encryptionConfig;
      if (!encryptionKeys.has(keyId)) {
        const cryptoKey = await importEncryptionKey(keyBytes);
        encryptionKeys.set(keyId, cryptoKey);
      }
      this._encryptionKeyId = keyId;
    }
    try {
      const opfsRoot2 = await navigator.storage.getDirectory();
      this._root = await opfsRoot2.getDirectoryHandle(`lanceql-${this.name}`, { create: true });
      this._ready = true;
    } catch (e) {
      console.error("[WorkerStore] Failed to open OPFS:", e);
      throw e;
    }
    return this;
  }
  _getCryptoKey() {
    return this._encryptionKeyId ? encryptionKeys.get(this._encryptionKeyId) : null;
  }
  async get(key) {
    await this._ensureOpen();
    try {
      const cryptoKey = this._getCryptoKey();
      const ext = cryptoKey ? ".enc" : ".json";
      const fileHandle = await this._root.getFileHandle(`${key}${ext}`);
      const file = await fileHandle.getFile();
      if (cryptoKey) {
        const buffer = await file.arrayBuffer();
        return decryptData(new Uint8Array(buffer), cryptoKey);
      } else {
        const text = await file.text();
        return JSON.parse(text);
      }
    } catch (e) {
      if (e.name === "NotFoundError") return void 0;
      throw e;
    }
  }
  async set(key, value) {
    await this._ensureOpen();
    const cryptoKey = this._getCryptoKey();
    const ext = cryptoKey ? ".enc" : ".json";
    const fileHandle = await this._root.getFileHandle(`${key}${ext}`, { create: true });
    const writable = await fileHandle.createWritable();
    if (cryptoKey) {
      const encrypted = await encryptData(value, cryptoKey);
      await writable.write(encrypted);
    } else {
      await writable.write(JSON.stringify(value));
    }
    await writable.close();
  }
  async delete(key) {
    await this._ensureOpen();
    const cryptoKey = this._getCryptoKey();
    const ext = cryptoKey ? ".enc" : ".json";
    try {
      await this._root.removeEntry(`${key}${ext}`);
    } catch (e) {
      if (e.name !== "NotFoundError") throw e;
    }
  }
  async keys() {
    await this._ensureOpen();
    const cryptoKey = this._getCryptoKey();
    const ext = cryptoKey ? ".enc" : ".json";
    const keys = [];
    for await (const [name] of this._root.entries()) {
      if (name.endsWith(ext)) {
        keys.push(name.slice(0, -ext.length));
      }
    }
    return keys;
  }
  async clear() {
    await this._ensureOpen();
    const entries = [];
    for await (const [name] of this._root.entries()) {
      entries.push(name);
    }
    for (const name of entries) {
      await this._root.removeEntry(name);
    }
  }
  async filter(key, query) {
    const value = await this.get(key);
    if (!Array.isArray(value)) {
      throw new Error(`Key '${key}' is not a collection`);
    }
    return value.filter((item) => this._matchQuery(item, query));
  }
  async find(key, query) {
    const value = await this.get(key);
    if (!Array.isArray(value)) {
      throw new Error(`Key '${key}' is not a collection`);
    }
    return value.find((item) => this._matchQuery(item, query));
  }
  async search(key, text, limit = 10) {
    const value = await this.get(key);
    if (!Array.isArray(value)) {
      throw new Error(`Key '${key}' is not a collection`);
    }
    if (value.length === 0) return [];
    if (this._embedder) {
      return this._semanticSearch(value, key, text, limit);
    }
    const textLower = text.toLowerCase();
    const scored = value.map((item) => {
      const itemText = this._extractText(item).toLowerCase();
      const words = textLower.split(/\s+/);
      const matchCount = words.filter((w) => itemText.includes(w)).length;
      return { item, score: matchCount / words.length };
    });
    return scored.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
  }
  async _semanticSearch(value, key, text, limit) {
    const queryVec = await this._embedder.embed(text);
    const scored = [];
    const textsToEmbed = [];
    const itemIndices = [];
    for (let i = 0; i < value.length; i++) {
      const item = value[i];
      const itemText = this._extractText(item);
      const cacheKey = `${this.name}:${key}:${itemText}`;
      if (embeddingCache.has(cacheKey)) {
        const cachedVec = embeddingCache.get(cacheKey);
        const score = this._cosineSimilarity(queryVec, cachedVec);
        scored.push({ item, score });
      } else {
        textsToEmbed.push(itemText);
        itemIndices.push(i);
      }
    }
    if (textsToEmbed.length > 0) {
      let itemVecs;
      if (textsToEmbed.length > 1 && this._embedder.embedBatch) {
        itemVecs = await this._embedder.embedBatch(textsToEmbed);
      } else {
        itemVecs = await Promise.all(textsToEmbed.map((t) => this._embedder.embed(t)));
      }
      for (let j = 0; j < itemVecs.length; j++) {
        const idx = itemIndices[j];
        const item = value[idx];
        const itemText = textsToEmbed[j];
        const itemVec = itemVecs[j];
        const cacheKey = `${this.name}:${key}:${itemText}`;
        embeddingCache.set(cacheKey, itemVec);
        const score = this._cosineSimilarity(queryVec, itemVec);
        scored.push({ item, score });
      }
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, limit);
  }
  async enableSemanticSearch(options = {}) {
    const { model = "minilm", onProgress } = options;
    if (!gpuTransformer) {
      if (!gpuTransformerPromise) {
        gpuTransformerPromise = this._initGPUTransformer();
        setGPUTransformerPromise(gpuTransformerPromise);
      }
      gpuTransformer = await gpuTransformerPromise;
      setGPUTransformer(gpuTransformer);
    }
    if (!gpuTransformer) {
      return null;
    }
    const modelConfig = await gpuTransformer.loadModel(model, onProgress);
    this._embedder = {
      model,
      dimensions: modelConfig.hiddenSize,
      embed: async (text) => gpuTransformer.encodeText(text, model),
      embedBatch: async (texts) => gpuTransformer.encodeTextBatch(texts, model)
    };
    return {
      model,
      dimensions: modelConfig.hiddenSize,
      type: modelConfig.modelType || "text"
    };
  }
  async _initGPUTransformer() {
    try {
      const webgpu = await import("./webgpu/index.js");
      if (!webgpu.isWebGPUAvailable()) {
        console.log("[WorkerStore] WebGPU not available");
        return null;
      }
      const transformer = webgpu.getGPUTransformer();
      const available = await transformer.init();
      if (!available) {
        console.log("[WorkerStore] WebGPU init failed");
        return null;
      }
      console.log("[WorkerStore] WebGPU initialized");
      return transformer;
    } catch (e) {
      console.error("[WorkerStore] WebGPU init error:", e);
      return null;
    }
  }
  disableSemanticSearch() {
    if (this._embedder && gpuTransformer) {
      gpuTransformer.unloadModel(this._embedder.model);
      this._embedder = null;
    }
  }
  hasSemanticSearch() {
    return this._embedder !== null;
  }
  async count(key, query = null) {
    const value = await this.get(key);
    if (!Array.isArray(value)) {
      throw new Error(`Key '${key}' is not a collection`);
    }
    if (!query) return value.length;
    return value.filter((item) => this._matchQuery(item, query)).length;
  }
  async _ensureOpen() {
    if (!this._ready) {
      await this.open();
    }
  }
  _matchQuery(item, query) {
    for (const [field, condition] of Object.entries(query)) {
      const value = item[field];
      if (typeof condition === "object" && condition !== null) {
        for (const [op, opVal] of Object.entries(condition)) {
          switch (op) {
            case "$eq":
              if (value !== opVal) return false;
              break;
            case "$ne":
              if (value === opVal) return false;
              break;
            case "$lt":
              if (!(value < opVal)) return false;
              break;
            case "$lte":
              if (!(value <= opVal)) return false;
              break;
            case "$gt":
              if (!(value > opVal)) return false;
              break;
            case "$gte":
              if (!(value >= opVal)) return false;
              break;
            case "$in":
              if (!Array.isArray(opVal) || !opVal.includes(value)) return false;
              break;
            case "$nin":
              if (Array.isArray(opVal) && opVal.includes(value)) return false;
              break;
            case "$contains":
              if (typeof value !== "string" || !value.includes(opVal)) return false;
              break;
            case "$regex":
              if (typeof value !== "string" || !new RegExp(opVal).test(value)) return false;
              break;
          }
        }
      } else {
        if (value !== condition) return false;
      }
    }
    return true;
  }
  _extractText(item) {
    if (typeof item === "string") return item;
    const texts = [];
    for (const [key, val] of Object.entries(item)) {
      if (typeof val === "string") texts.push(val);
    }
    return texts.join(" ");
  }
  _cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
};

// src/worker/opfs-storage.js
var OPFSStorage = class {
  constructor(rootDir = "lanceql") {
    this.rootDir = rootDir;
    this.root = null;
  }
  async getRoot() {
    if (this.root) return this.root;
    if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
      throw new Error("OPFS not available");
    }
    const opfsRoot2 = await navigator.storage.getDirectory();
    this.root = await opfsRoot2.getDirectoryHandle(this.rootDir, { create: true });
    return this.root;
  }
  async open() {
    await this.getRoot();
    return this;
  }
  async getDir(path) {
    const root = await this.getRoot();
    const parts = path.split("/").filter((p) => p);
    let current = root;
    for (const part of parts) {
      current = await current.getDirectoryHandle(part, { create: true });
    }
    return current;
  }
  async save(path, data) {
    const parts = path.split("/");
    const fileName = parts.pop();
    const dirPath = parts.join("/");
    const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
    const fileHandle = await dir.getFileHandle(fileName, { create: true });
    if (fileHandle.createSyncAccessHandle) {
      try {
        const accessHandle = await fileHandle.createSyncAccessHandle();
        accessHandle.truncate(0);
        accessHandle.write(data, { at: 0 });
        accessHandle.flush();
        accessHandle.close();
        return { path, size: data.byteLength };
      } catch (e) {
      }
    }
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
    return { path, size: data.byteLength };
  }
  async load(path) {
    try {
      const parts = path.split("/");
      const fileName = parts.pop();
      const dirPath = parts.join("/");
      const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
      const fileHandle = await dir.getFileHandle(fileName);
      const file = await fileHandle.getFile();
      const buffer = await file.arrayBuffer();
      return new Uint8Array(buffer);
    } catch (e) {
      if (e.name === "NotFoundError") return null;
      throw e;
    }
  }
  async delete(path) {
    try {
      const parts = path.split("/");
      const fileName = parts.pop();
      const dirPath = parts.join("/");
      const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
      await dir.removeEntry(fileName);
      return true;
    } catch (e) {
      if (e.name === "NotFoundError") return false;
      throw e;
    }
  }
  async list(dirPath = "") {
    try {
      const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
      const files = [];
      for await (const [name, handle] of dir.entries()) {
        files.push({ name, type: handle.kind });
      }
      return files;
    } catch (e) {
      return [];
    }
  }
  async exists(path) {
    try {
      const parts = path.split("/");
      const fileName = parts.pop();
      const dirPath = parts.join("/");
      const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
      await dir.getFileHandle(fileName);
      return true;
    } catch (e) {
      return false;
    }
  }
  async deleteDir(dirPath) {
    try {
      const parts = dirPath.split("/");
      const dirName = parts.pop();
      const parentPath = parts.join("/");
      const parent = parentPath ? await this.getDir(parentPath) : await this.getRoot();
      await parent.removeEntry(dirName, { recursive: true });
      return true;
    } catch (e) {
      return false;
    }
  }
};
var opfsStorage = new OPFSStorage();

// src/worker/data-types.js
var E = new TextEncoder();
var D = new TextDecoder();
var DataType = {
  INT64: "int64",
  INT32: "int32",
  FLOAT64: "float64",
  FLOAT32: "float32",
  STRING: "string",
  BOOL: "bool",
  VECTOR: "vector"
};
var TYPE_INT32 = 1;
var TYPE_INT64 = 2;
var TYPE_FLOAT32 = 3;
var TYPE_FLOAT64 = 4;
var TYPE_STRING = 5;
var TYPE_BOOL = 6;
function getTypedArrayForType(dataType) {
  switch (dataType) {
    case "int32":
    case "integer":
      return Int32Array;
    case "int64":
      return BigInt64Array;
    case "float32":
    case "real":
      return Float32Array;
    case "float64":
    case "double":
      return Float64Array;
    default:
      return null;
  }
}
function getTypeCode(dataType) {
  switch (dataType) {
    case "int32":
    case "integer":
      return TYPE_INT32;
    case "int64":
      return TYPE_INT64;
    case "float32":
    case "real":
      return TYPE_FLOAT32;
    case "float64":
    case "double":
      return TYPE_FLOAT64;
    case "string":
    case "text":
      return TYPE_STRING;
    case "bool":
    case "boolean":
      return TYPE_BOOL;
    default:
      return TYPE_STRING;
  }
}
var LanceFileWriter = class {
  constructor(schema) {
    this.schema = schema;
    this.columns = /* @__PURE__ */ new Map();
    this.rowCount = 0;
    this._useBinary = true;
  }
  addRows(rows) {
    if (rows.length === 0) return;
    if (this.rowCount === 0) {
      for (const col of this.schema) {
        const TypedArray = getTypedArrayForType(col.dataType);
        if (TypedArray) {
          this.columns.set(col.name, {
            type: "typed",
            dataType: col.dataType,
            data: new TypedArray(Math.max(rows.length, 1024)),
            length: 0
          });
        } else {
          this.columns.set(col.name, {
            type: "array",
            dataType: col.dataType,
            data: [],
            length: 0
          });
        }
      }
    }
    for (const col of this.schema) {
      const column = this.columns.get(col.name);
      if (column.type === "typed") {
        const newLength = column.length + rows.length;
        if (newLength > column.data.length) {
          const newSize = Math.max(newLength, column.data.length * 2);
          const newData = new column.data.constructor(newSize);
          newData.set(column.data);
          column.data = newData;
        }
        for (let i = 0; i < rows.length; i++) {
          const val = rows[i][col.name];
          column.data[column.length + i] = val ?? 0;
        }
        column.length += rows.length;
      } else {
        for (let i = 0; i < rows.length; i++) {
          column.data.push(rows[i][col.name] ?? null);
        }
        column.length += rows.length;
      }
    }
    this.rowCount += rows.length;
  }
  /**
   * Build binary columnar format
   * Format: [magic(4)] [version(4)] [numCols(4)] [rowCount(4)]
   *         [colName1Len(4)] [colName1] [colType(1)] [colDataLen(4)] [colData]
   *         ...
   */
  buildBinary() {
    const chunks = [];
    let totalSize = 16;
    const columnChunks = [];
    for (const col of this.schema) {
      const column = this.columns.get(col.name);
      const nameBytes = E.encode(col.name);
      const typeCode = getTypeCode(col.dataType);
      let dataBytes;
      if (column.type === "typed") {
        const trimmed = column.data.subarray(0, column.length);
        dataBytes = new Uint8Array(trimmed.buffer, trimmed.byteOffset, trimmed.byteLength);
      } else {
        dataBytes = E.encode(JSON.stringify(column.data));
      }
      columnChunks.push({ nameBytes, typeCode, dataBytes, dataType: col.dataType });
      const currentOffsetWithoutPadding = totalSize + 4 + nameBytes.length + 1;
      const padding = (8 - (currentOffsetWithoutPadding + 4) % 8) % 8;
      totalSize += 4 + nameBytes.length + 1 + padding + 4 + dataBytes.length;
    }
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;
    view.setUint32(offset, 1279348291, false);
    offset += 4;
    view.setUint32(offset, 2, false);
    offset += 4;
    view.setUint32(offset, this.schema.length, false);
    offset += 4;
    view.setUint32(offset, this.rowCount, false);
    offset += 4;
    for (const chunk of columnChunks) {
      view.setUint32(offset, chunk.nameBytes.length, false);
      offset += 4;
      bytes.set(chunk.nameBytes, offset);
      offset += chunk.nameBytes.length;
      view.setUint8(offset, chunk.typeCode);
      offset += 1;
      const padding = (8 - (offset + 4) % 8) % 8;
      for (let i = 0; i < padding; i++) {
        view.setUint8(offset + i, 0);
      }
      offset += padding;
      view.setUint32(offset, chunk.dataBytes.length, false);
      offset += 4;
      bytes.set(chunk.dataBytes, offset);
      offset += chunk.dataBytes.length;
    }
    return new Uint8Array(buffer);
  }
  /**
   * Set columnar data directly (no row conversion needed)
   * @param {Object} columnarData - { colName: Array }
   */
  setColumnarData(columnarData) {
    const firstCol = Object.keys(columnarData)[0];
    this.rowCount = columnarData[firstCol]?.length || 0;
    for (const col of this.schema) {
      const arr = columnarData[col.name];
      if (!arr) continue;
      const TypedArray = getTypedArrayForType(col.dataType);
      if (TypedArray && ArrayBuffer.isView(arr)) {
        this.columns.set(col.name, {
          type: "typed",
          dataType: col.dataType,
          data: arr,
          length: arr.length
        });
      } else if (TypedArray) {
        const typedArr = new TypedArray(arr.length);
        for (let i = 0; i < arr.length; i++) {
          typedArr[i] = arr[i] ?? 0;
        }
        this.columns.set(col.name, {
          type: "typed",
          dataType: col.dataType,
          data: typedArr,
          length: arr.length
        });
      } else {
        this.columns.set(col.name, {
          type: "array",
          dataType: col.dataType,
          data: Array.isArray(arr) ? arr : Array.from(arr),
          length: arr.length
        });
      }
    }
  }
  build() {
    if (this._useBinary && this.rowCount > 0) {
      try {
        return this.buildBinary();
      } catch (e) {
        console.warn("[LanceFileWriter] Binary build failed, falling back to JSON:", e);
      }
    }
    const data = {
      format: "json",
      schema: this.schema,
      columns: {},
      rowCount: this.rowCount
    };
    for (const [name, column] of this.columns) {
      if (column.type === "typed") {
        data.columns[name] = Array.from(column.data.subarray(0, column.length));
      } else {
        data.columns[name] = column.data;
      }
    }
    return E.encode(JSON.stringify(data));
  }
};
function parseBinaryColumnar(buffer) {
  const view = new DataView(buffer.buffer || buffer);
  const bytes = new Uint8Array(buffer.buffer || buffer);
  let offset = 0;
  const magic = view.getUint32(offset, false);
  if (magic !== 1279348291) {
    return null;
  }
  offset += 4;
  const version = view.getUint32(offset, false);
  offset += 4;
  const numCols = view.getUint32(offset, false);
  offset += 4;
  const rowCount = view.getUint32(offset, false);
  offset += 4;
  const schema = [];
  const columns = {};
  for (let i = 0; i < numCols; i++) {
    const nameLen = view.getUint32(offset, false);
    offset += 4;
    const name = D.decode(bytes.subarray(offset, offset + nameLen));
    offset += nameLen;
    const typeCode = view.getUint8(offset);
    offset += 1;
    const padding = (8 - (offset + 4) % 8) % 8;
    offset += padding;
    const dataLen = view.getUint32(offset, false);
    offset += 4;
    const dataBytes = bytes.subarray(offset, offset + dataLen);
    offset += dataLen;
    let data, dataType;
    try {
      switch (typeCode) {
        case TYPE_INT32:
          dataType = "int32";
          if (dataBytes.byteOffset % 4 !== 0) {
            data = new Int32Array(dataBytes.slice().buffer);
          } else {
            data = new Int32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
          }
          break;
        case TYPE_FLOAT32:
          dataType = "float32";
          if (dataBytes.byteOffset % 4 !== 0) {
            data = new Float32Array(dataBytes.slice().buffer);
          } else {
            data = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
          }
          break;
        case TYPE_FLOAT64:
          dataType = "float64";
          if (dataBytes.byteOffset % 8 !== 0) {
            data = new Float64Array(dataBytes.slice().buffer);
          } else {
            data = new Float64Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 8);
          }
          break;
        case TYPE_INT64:
          dataType = "int64";
          if (dataBytes.byteOffset % 8 !== 0) {
            data = new BigInt64Array(dataBytes.slice().buffer);
          } else {
            data = new BigInt64Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 8);
          }
          break;
        case TYPE_STRING:
        case TYPE_BOOL:
        default:
          dataType = typeCode === TYPE_BOOL ? "bool" : "string";
          data = JSON.parse(D.decode(dataBytes));
          break;
      }
    } catch (e) {
      console.error(`[LanceQLWorker] Error parsing column '${name}' (type ${typeCode}, len ${dataLen}):`, e);
      throw e;
    }
    schema.push({ name, dataType });
    columns[name] = data;
  }
  return { schema, columns, rowCount, format: "binary" };
}

// src/worker/buffer-pool.js
var BufferPool = class {
  /**
   * @param {number} maxBytes - Maximum memory in bytes (default 512MB)
   */
  constructor(maxBytes = 512 * 1024 * 1024) {
    this.maxBytes = maxBytes;
    this.currentBytes = 0;
    this.cache = /* @__PURE__ */ new Map();
    this.head = null;
    this.tail = null;
  }
  /**
   * Get an item from the pool. Updates recency.
   * @param {string} key
   * @returns {any} value or undefined
   */
  get(key) {
    const node = this.cache.get(key);
    if (!node) return void 0;
    this._moveToHead(node);
    return node.value;
  }
  /**
   * Add or update an item in the pool. Evicts if necessary.
   * @param {string} key
   * @param {any} value
   * @param {number} size - Approximate size in bytes
   */
  set(key, value, size = 0) {
    const existingNode = this.cache.get(key);
    if (existingNode) {
      this.currentBytes -= existingNode.size;
      this.currentBytes += size;
      existingNode.value = value;
      existingNode.size = size;
      this._moveToHead(existingNode);
    } else {
      const newNode = { key, value, size, prev: null, next: null };
      this.cache.set(key, newNode);
      this._addToHead(newNode);
      this.currentBytes += size;
    }
    this._evictIfNeeded();
  }
  /**
   * Check if a key exists without updating recency.
   * @param {string} key
   */
  has(key) {
    return this.cache.has(key);
  }
  /**
   * Remove an item from the pool.
   * @param {string} key
   */
  delete(key) {
    const node = this.cache.get(key);
    if (node) {
      this._removeNode(node);
      this.cache.delete(key);
      this.currentBytes -= node.size;
    }
  }
  /**
   * Clear the pool.
   */
  clear() {
    this.cache.clear();
    this.head = null;
    this.tail = null;
    this.currentBytes = 0;
  }
  _moveToHead(node) {
    if (node === this.head) return;
    this._removeNode(node);
    this._addToHead(node);
  }
  _addToHead(node) {
    node.next = this.head;
    node.prev = null;
    if (this.head) {
      this.head.prev = node;
    }
    this.head = node;
    if (!this.tail) {
      this.tail = node;
    }
  }
  _removeNode(node) {
    if (node.prev) {
      node.prev.next = node.next;
    } else {
      this.head = node.next;
    }
    if (node.next) {
      node.next.prev = node.prev;
    } else {
      this.tail = node.prev;
    }
  }
  _evictIfNeeded() {
    while (this.currentBytes > this.maxBytes && this.tail) {
      const node = this.tail;
      this._removeNode(node);
      this.cache.delete(node.key);
      this.currentBytes -= node.size;
    }
  }
  /**
   * Helper to estimate size of common objects
   * @param {any} val
   * @returns {number}
   */
  static estimateSize(val) {
    if (!val) return 0;
    if (val.byteLength) return val.byteLength;
    if (Array.isArray(val)) {
      return val.length * 100;
    }
    if (val.buffer && val.buffer.byteLength) return val.buffer.byteLength;
    return 1e3;
  }
};

// src/worker/worker-database.js
var scanStreams = /* @__PURE__ */ new Map();
var nextScanId = 1;
var WorkerDatabase = class {
  constructor(name, bufferPool2) {
    this.name = name;
    this.tables = /* @__PURE__ */ new Map();
    this.version = 0;
    this.manifestKey = `${name}/__manifest__`;
    this.bufferPool = bufferPool2 || new BufferPool();
    this._writeBuffer = /* @__PURE__ */ new Map();
    this._flushTimer = null;
    this._flushInterval = 2e3;
    this._flushThreshold = 1e4;
    this._columnarBuffer = /* @__PURE__ */ new Map();
    this._flushing = false;
    this._readCache = /* @__PURE__ */ new Map();
    this._columnarCache = /* @__PURE__ */ new Map();
  }
  async open() {
    const manifestData = await opfsStorage.load(this.manifestKey);
    if (manifestData) {
      const manifest = JSON.parse(D.decode(manifestData));
      this.version = manifest.version || 0;
      this.tables = new Map(Object.entries(manifest.tables || {}));
    }
    return this;
  }
  async _saveManifest() {
    this.version++;
    const manifest = {
      version: this.version,
      timestamp: Date.now(),
      tables: Object.fromEntries(this.tables)
    };
    const data = E.encode(JSON.stringify(manifest));
    await opfsStorage.save(this.manifestKey, data);
  }
  async createTable(tableName, columns, ifNotExists = false) {
    if (this.tables.has(tableName)) {
      if (ifNotExists) {
        return { success: true, existed: true };
      }
      throw new Error(`Table '${tableName}' already exists`);
    }
    const schema = columns.map((col) => ({
      name: col.name,
      type: DataType[(col.dataType || col.type)?.toUpperCase()] || col.dataType || col.type || "string",
      primaryKey: col.primaryKey || false,
      vectorDim: col.vectorDim || null
    }));
    const tableState = {
      name: tableName,
      schema,
      fragments: [],
      deletionVector: [],
      rowCount: 0,
      nextRowId: 0,
      version: 0,
      createdAt: Date.now()
    };
    this.tables.set(tableName, tableState);
    await this._saveManifest();
    return { success: true, table: tableName };
  }
  async dropTable(tableName, ifExists = false) {
    if (!this.tables.has(tableName)) {
      if (ifExists) {
        this._writeBuffer.delete(tableName);
        return { success: true, existed: false };
      }
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const table = this.tables.get(tableName);
    this._writeBuffer.delete(tableName);
    for (const fragKey of table.fragments) {
      this._readCache.delete(fragKey);
      await opfsStorage.delete(fragKey);
    }
    this.tables.delete(tableName);
    await this._saveManifest();
    return { success: true, table: tableName };
  }
  async insert(tableName, rows) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const table = this.tables.get(tableName);
    const n = rows.length;
    if (!this._columnarBuffer.has(tableName)) {
      const initialCapacity = Math.max(1024, n * 2);
      const cols = {
        __rowId: new Float64Array(initialCapacity),
        __length: 0,
        __capacity: initialCapacity,
        __schema: table.schema
      };
      for (const c of table.schema) {
        const type = (c.dataType || c.type || "").toLowerCase();
        if (type === "text" || type === "string" || type === "varchar") {
          cols[c.name] = new Array(initialCapacity);
        } else if (type === "int64" || type === "bigint") {
          cols[c.name] = new BigInt64Array(initialCapacity);
        } else {
          cols[c.name] = new Float64Array(initialCapacity);
        }
      }
      this._columnarBuffer.set(tableName, cols);
    }
    const colBuf = this._columnarBuffer.get(tableName);
    let len = colBuf.__length;
    let cap = colBuf.__capacity;
    if (len + n > cap) {
      const newCap = Math.max(cap * 2, len + n);
      const newRowId = new Float64Array(newCap);
      newRowId.set(colBuf.__rowId.subarray(0, len));
      colBuf.__rowId = newRowId;
      for (const c of table.schema) {
        const old = colBuf[c.name];
        if (old instanceof Float64Array) {
          const newArr = new Float64Array(newCap);
          newArr.set(old.subarray(0, len));
          colBuf[c.name] = newArr;
        } else if (old instanceof BigInt64Array) {
          const newArr = new BigInt64Array(newCap);
          newArr.set(old.subarray(0, len));
          colBuf[c.name] = newArr;
        } else {
          colBuf[c.name].length = newCap;
        }
      }
      colBuf.__capacity = newCap;
    }
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      colBuf.__rowId[len + i] = table.nextRowId++;
      for (const c of table.schema) {
        let val = row[c.name];
        if (colBuf[c.name] instanceof Float64Array) {
          colBuf[c.name][len + i] = val !== null && val !== void 0 ? Number(val) : NaN;
        } else if (colBuf[c.name] instanceof BigInt64Array) {
          colBuf[c.name][len + i] = val !== null && val !== void 0 ? BigInt(val) : 0n;
        } else {
          colBuf[c.name][len + i] = val ?? null;
        }
      }
    }
    colBuf.__length = len + n;
    table.rowCount += n;
    table.version = (table.version || 0) + 1;
    this._scheduleFlush();
    if (colBuf.__length >= this._flushThreshold) {
      await this._flushTable(tableName);
    }
    return { success: true, inserted: n };
  }
  _scheduleFlush() {
    if (this._flushTimer) return;
    this._flushTimer = setTimeout(() => {
      this._flushTimer = null;
      this.flush().catch((e) => console.warn("[WorkerDatabase] Flush error:", e));
    }, this._flushInterval);
  }
  async flush() {
    if (this._flushing) return;
    this._flushing = true;
    try {
      const tables = [...this._columnarBuffer.keys()];
      for (const tableName of tables) {
        await this._flushTable(tableName);
      }
    } finally {
      this._flushing = false;
    }
  }
  async _flushTable(tableName) {
    const colBuf = this._columnarBuffer.get(tableName);
    const bufLen = colBuf?.__length || 0;
    if (!colBuf || bufLen === 0) return;
    const table = this.tables.get(tableName);
    if (!table) return;
    const schemaWithRowId = [
      { name: "__rowId", type: "int64", dataType: "float64", primaryKey: true },
      ...table.schema.filter((c) => c.name !== "__rowId").map((c) => {
        const type = (c.dataType || c.type || "").toLowerCase();
        if (type === "int64" || type === "bigint") {
          return { ...c, dataType: "int64" };
        }
        const isNumeric = type === "float64" || type === "float32" || type === "int32" || type === "integer" || type === "real" || type === "double";
        return {
          ...c,
          dataType: isNumeric ? "float64" : c.dataType || c.type || "float64"
        };
      })
    ];
    const columnarData = {};
    for (const col of schemaWithRowId) {
      const arr = colBuf[col.name];
      if (!arr) continue;
      if (arr instanceof Float64Array || arr instanceof BigInt64Array) {
        columnarData[col.name] = arr.subarray(0, bufLen);
      } else {
        columnarData[col.name] = arr.slice(0, bufLen);
      }
    }
    colBuf.__length = 0;
    const writer = new LanceFileWriter(schemaWithRowId);
    writer.setColumnarData(columnarData);
    const lanceData = writer.build();
    const fragKey = `${this.name}/${tableName}/frag_${Date.now()}_${Math.random().toString(36).slice(2)}.lance`;
    await opfsStorage.save(fragKey, lanceData);
    table.fragments.push(fragKey);
    await this._saveManifest();
  }
  async delete(tableName, predicateFn) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const table = this.tables.get(tableName);
    let deletedCount = 0;
    const colBuf = this._columnarBuffer.get(tableName);
    const bufLen = colBuf?.__length || 0;
    if (colBuf && bufLen > 0) {
      const colNames = table.schema.map((c) => c.name);
      for (let i = 0; i < bufLen; i++) {
        const row = { __rowId: colBuf.__rowId[i] };
        for (const name of colNames) {
          const v = colBuf[name][i];
          row[name] = Number.isNaN(v) ? null : v;
        }
        if (predicateFn(row)) {
          if (!table.deletionVector.includes(colBuf.__rowId[i])) {
            table.deletionVector.push(colBuf.__rowId[i]);
            deletedCount++;
          }
        }
      }
    }
    for (const fragKey of table.fragments) {
      const fragData = await opfsStorage.load(fragKey);
      if (fragData) {
        const rows = this._parseFragment(fragData, table.schema);
        for (const row of rows) {
          if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
            table.deletionVector.push(row.__rowId);
            deletedCount++;
          }
        }
      }
    }
    table.rowCount -= deletedCount;
    table.version = (table.version || 0) + 1;
    await this._saveManifest();
    return { success: true, deleted: deletedCount };
  }
  async update(tableName, updates, predicateFn) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const table = this.tables.get(tableName);
    let updatedCount = 0;
    const buffer = this._writeBuffer.get(tableName);
    if (buffer && buffer.length > 0) {
      for (const row of buffer) {
        if (predicateFn(row)) {
          Object.assign(row, updates);
          updatedCount++;
        }
      }
    }
    const persistedUpdates = [];
    for (const fragKey of table.fragments) {
      const fragData = await opfsStorage.load(fragKey);
      if (fragData) {
        const rows = this._parseFragment(fragData, table.schema);
        for (const row of rows) {
          if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
            table.deletionVector.push(row.__rowId);
            table.rowCount--;
            const newRow = { ...row, ...updates };
            delete newRow.__rowId;
            persistedUpdates.push(newRow);
            updatedCount++;
          }
        }
      }
    }
    if (persistedUpdates.length > 0) {
      await this.insert(tableName, persistedUpdates);
    } else {
      await this._saveManifest();
    }
    return { success: true, updated: updatedCount };
  }
  async updateWithExpr(tableName, updateExprs, predicateFn, evalExpr) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const table = this.tables.get(tableName);
    let updatedCount = 0;
    const colBuf = this._columnarBuffer.get(tableName);
    const bufLen = colBuf?.__length || 0;
    if (colBuf && bufLen > 0) {
      const colNames = table.schema.map((c) => c.name);
      for (let i = 0; i < bufLen; i++) {
        const row = { __rowId: colBuf.__rowId[i] };
        for (const name of colNames) {
          const v = colBuf[name][i];
          row[name] = Number.isNaN(v) ? null : v;
        }
        if (predicateFn(row)) {
          for (const [col, expr] of Object.entries(updateExprs)) {
            const newVal = evalExpr(expr, row);
            if (colBuf[col] !== void 0) {
              colBuf[col][i] = newVal ?? (colBuf[col] instanceof Float64Array ? NaN : null);
            }
          }
          table.version = (table.version || 0) + 1;
          updatedCount++;
        }
      }
    }
    const persistedUpdates = [];
    for (const fragKey of table.fragments) {
      const fragData = await opfsStorage.load(fragKey);
      if (fragData) {
        const rows = this._parseFragment(fragData, table.schema);
        for (const row of rows) {
          if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
            table.deletionVector.push(row.__rowId);
            table.rowCount--;
            const newRow = { ...row };
            for (const [col, expr] of Object.entries(updateExprs)) {
              newRow[col] = evalExpr(expr, row);
            }
            delete newRow.__rowId;
            persistedUpdates.push(newRow);
            updatedCount++;
          }
        }
      }
    }
    if (persistedUpdates.length > 0) {
      await this.insert(tableName, persistedUpdates);
    } else {
      await this._saveManifest();
    }
    return { success: true, updated: updatedCount };
  }
  async select(tableName, options = {}) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    let rows = await this._readAllRows(tableName);
    if (options.where) {
      rows = rows.filter(options.where);
    }
    if (options.orderBy) {
      const { column, desc } = options.orderBy;
      rows.sort((a, b) => {
        const cmp = a[column] < b[column] ? -1 : a[column] > b[column] ? 1 : 0;
        return desc ? -cmp : cmp;
      });
    }
    if (options.offset) {
      rows = rows.slice(options.offset);
    }
    if (options.limit) {
      rows = rows.slice(0, options.limit);
    }
    const projectCols = options.columns && options.columns.length > 0 && options.columns[0] !== "*" ? options.columns : null;
    const result = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (projectCols) {
        const projected = {};
        for (const col of projectCols) {
          projected[col] = row[col];
        }
        result[i] = projected;
      } else {
        const { __rowId, ...rest } = row;
        result[i] = rest;
      }
    }
    return result;
  }
  async _readAllRows(tableName) {
    const table = this.tables.get(tableName);
    const colBuf = this._columnarBuffer.get(tableName);
    const hasDeleted = table.deletionVector.length > 0;
    const deletedSet = hasDeleted ? new Set(table.deletionVector) : null;
    const allRows = [];
    for (const fragKey of table.fragments) {
      let binary = this.bufferPool.get(fragKey);
      let rows = null;
      if (!binary) {
        const fragData = await opfsStorage.load(fragKey);
        if (fragData) {
          binary = parseBinaryColumnar(fragData);
          if (binary) {
            this.bufferPool.set(fragKey, binary, fragData.byteLength);
          } else {
            rows = this._parseFragment(fragData, table.schema);
          }
        }
      }
      if (binary && !rows) {
        rows = this._hydrateRowsFromBinary(binary, table.schema);
      }
      rows = rows || [];
      if (!deletedSet) {
        allRows.push(...rows);
      } else {
        for (const row of rows) {
          if (!deletedSet.has(row.__rowId)) {
            allRows.push(row);
          }
        }
      }
    }
    const bufLen = colBuf?.__length || 0;
    if (colBuf && bufLen > 0) {
      const colNames = table.schema.map((c) => c.name);
      for (let i = 0; i < bufLen; i++) {
        if (deletedSet && deletedSet.has(colBuf.__rowId[i])) continue;
        const row = { __rowId: colBuf.__rowId[i] };
        for (const name of colNames) {
          const v = colBuf[name][i];
          row[name] = Number.isNaN(v) ? null : v;
        }
        allRows.push(row);
      }
    }
    return allRows;
  }
  _hydrateRowsFromBinary(binary, schema) {
    const { columns, rowCount } = binary;
    const colNames = schema.map((c) => c.name);
    const rows = new Array(rowCount);
    for (let i = 0; i < rowCount; i++) {
      const row = { __rowId: columns.__rowId[i] };
      for (const name of colNames) {
        if (columns[name]) {
          row[name] = columns[name][i];
        }
      }
      rows[i] = row;
    }
    return rows;
  }
  _parseFragment(data, schema) {
    try {
      const binary = parseBinaryColumnar(data);
      if (binary) {
        return this._parseBinaryColumnar(binary);
      }
      const text = D.decode(data);
      const parsed = JSON.parse(text);
      if (parsed.format === "json" && parsed.columns) {
        return this._parseJsonColumnar(parsed);
      }
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch (e) {
      console.warn("[WorkerDatabase] Failed to parse fragment:", e);
      return [];
    }
  }
  /**
   * Select data in COLUMNAR format (no row conversion).
   * Returns { schema, columns: { colName: TypedArray }, rowCount }
   */
  async selectColumnar(tableName) {
    const table = this.tables.get(tableName);
    if (!table) return null;
    const bufLen = this._columnarBuffer.get(tableName)?.__length || 0;
    const version = `${table.fragments.length}:${bufLen}:${table.deletionVector.length}:${table.version || 0}`;
    const cached = this._columnarCache.get(tableName);
    if (cached && cached.version === version) {
      const columns = {};
      for (const [name, arr] of Object.entries(cached.data.columns)) {
        if (ArrayBuffer.isView(arr)) {
          columns[name] = new arr.constructor(arr.buffer, arr.byteOffset, arr.length);
        } else {
          columns[name] = arr;
        }
      }
      return { schema: table.schema, columns, rowCount: cached.data.rowCount };
    }
    const hasDeleted = table.deletionVector.length > 0;
    const deletedSet = hasDeleted ? new Set(table.deletionVector) : null;
    const allColumns = {};
    const colNames = table.schema.map((c) => c.name);
    for (const name of colNames) allColumns[name] = [];
    allColumns.__rowId = [];
    for (const fragKey of table.fragments) {
      let binary = this.bufferPool.get(fragKey);
      if (!binary) {
        const fragData = await opfsStorage.load(fragKey);
        if (!fragData) continue;
        binary = parseBinaryColumnar(fragData);
        if (binary) {
          this.bufferPool.set(fragKey, binary, fragData.byteLength);
        }
      }
      if (!binary) continue;
      const { columns, rowCount } = binary;
      if (!deletedSet) {
        for (const name of colNames) {
          if (columns[name]) allColumns[name].push(columns[name]);
        }
        if (columns.__rowId) allColumns.__rowId.push(columns.__rowId);
      } else {
        const rowIds = columns.__rowId;
        const validIndices = [];
        for (let i = 0; i < rowCount; i++) {
          if (!deletedSet.has(rowIds[i])) validIndices.push(i);
        }
        for (const name of colNames) {
          if (columns[name]) {
            const arr = columns[name];
            const filtered = new arr.constructor(validIndices.length);
            for (let j = 0; j < validIndices.length; j++) {
              filtered[j] = arr[validIndices[j]];
            }
            allColumns[name].push(filtered);
          }
        }
      }
    }
    const colBuf = this._columnarBuffer.get(tableName);
    const bufLen2 = colBuf?.__length || 0;
    if (colBuf && bufLen2 > 0) {
      if (!deletedSet) {
        for (const col of table.schema) {
          const arr = colBuf[col.name];
          if (!arr) continue;
          if (arr instanceof Float64Array) {
            const owned = new Float64Array(bufLen2);
            owned.set(arr.subarray(0, bufLen2));
            allColumns[col.name].push(owned);
          } else {
            allColumns[col.name].push(arr.slice(0, bufLen2));
          }
        }
        const ownedRowId = new Float64Array(bufLen2);
        ownedRowId.set(colBuf.__rowId.subarray(0, bufLen2));
        allColumns.__rowId.push(ownedRowId);
      } else {
        const validIndices = [];
        for (let i = 0; i < bufLen2; i++) {
          if (!deletedSet.has(colBuf.__rowId[i])) validIndices.push(i);
        }
        const vLen = validIndices.length;
        for (const col of table.schema) {
          const arr = colBuf[col.name];
          if (!arr) continue;
          if (arr instanceof Float64Array) {
            const filtered = new Float64Array(vLen);
            for (let j = 0; j < vLen; j++) filtered[j] = arr[validIndices[j]];
            allColumns[col.name].push(filtered);
          } else {
            allColumns[col.name].push(validIndices.map((i) => arr[i]));
          }
        }
        const filteredRowId = new Float64Array(vLen);
        for (let j = 0; j < vLen; j++) filteredRowId[j] = colBuf.__rowId[validIndices[j]];
        allColumns.__rowId.push(filteredRowId);
      }
    }
    const mergedColumns = {};
    let totalRows = 0;
    for (const name of [...colNames, "__rowId"]) {
      const arrays = allColumns[name];
      if (arrays.length === 0) {
        mergedColumns[name] = new Float64Array(0);
      } else if (arrays.length === 1) {
        mergedColumns[name] = arrays[0];
        if (totalRows === 0) totalRows = arrays[0].length;
      } else {
        const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
        if (totalRows === 0) totalRows = totalLen;
        const first = arrays[0];
        const merged = ArrayBuffer.isView(first) ? new first.constructor(totalLen) : new Array(totalLen);
        let offset = 0;
        for (const arr of arrays) {
          if (ArrayBuffer.isView(merged)) {
            merged.set(arr, offset);
          } else {
            for (let i = 0; i < arr.length; i++) merged[offset + i] = arr[i];
          }
          offset += arr.length;
        }
        mergedColumns[name] = merged;
      }
    }
    const result = { schema: table.schema, columns: mergedColumns, rowCount: totalRows };
    this._columnarCache.set(tableName, { version, data: result });
    return result;
  }
  /**
   * Read column data directly (no row conversion) for fast aggregation.
   * Returns typed array for numeric columns.
   */
  async _readColumn(tableName, colName) {
    const table = this.tables.get(tableName);
    if (!table) return null;
    const buffer = this._writeBuffer.get(tableName);
    const colData = [];
    for (const fragKey of table.fragments) {
      const fragData = await opfsStorage.load(fragKey);
      if (!fragData) continue;
      const binary = parseBinaryColumnar(fragData);
      if (binary && binary.columns[colName]) {
        const arr = binary.columns[colName];
        if (arr.length > 0) colData.push(arr);
      }
    }
    if (buffer && buffer.length > 0) {
      const bufCol = new Float64Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        const v = buffer[i][colName];
        bufCol[i] = typeof v === "number" ? v : 0;
      }
      colData.push(bufCol);
    }
    if (colData.length === 0) return new Float64Array(0);
    if (colData.length === 1) return colData[0];
    const totalLen = colData.reduce((sum, arr) => sum + arr.length, 0);
    const merged = new Float64Array(totalLen);
    let offset = 0;
    for (const arr of colData) {
      merged.set(arr, offset);
      offset += arr.length;
    }
    return merged;
  }
  /**
   * Parse binary columnar format - optimized for typed arrays
   */
  _parseBinaryColumnar(data) {
    const { schema, columns, rowCount } = data;
    const rows = new Array(rowCount);
    const colNames = schema.map((c) => c.name);
    const colArrays = colNames.map((name) => columns[name]);
    const numCols = colNames.length;
    for (let i = 0; i < rowCount; i++) {
      const row = {};
      for (let j = 0; j < numCols; j++) {
        row[colNames[j]] = colArrays[j][i] ?? null;
      }
      rows[i] = row;
    }
    return rows;
  }
  _parseJsonColumnar(data) {
    const { schema, columns, rowCount } = data;
    const rows = new Array(rowCount);
    const colNames = schema.map((c) => c.name);
    const colArrays = colNames.map((name) => columns[name] || []);
    const numCols = colNames.length;
    for (let i = 0; i < rowCount; i++) {
      const row = {};
      for (let j = 0; j < numCols; j++) {
        row[colNames[j]] = colArrays[j][i] ?? null;
      }
      rows[i] = row;
    }
    return rows;
  }
  getTable(tableName) {
    return this.tables.get(tableName);
  }
  listTables() {
    return Array.from(this.tables.keys());
  }
  /**
   * Get fragment paths for direct WASM access (no row conversion)
   */
  getFragmentPaths(tableName) {
    const table = this.tables.get(tableName);
    if (!table) return [];
    return table.fragments;
  }
  /**
   * Get column index by name for a table
   */
  getColumnIndex(tableName, colName) {
    const table = this.tables.get(tableName);
    if (!table) return -1;
    const idx = table.schema.findIndex((c) => c.name === colName);
    return idx >= 0 ? idx + 1 : -1;
  }
  /**
   * Check if table has buffered (unflushed) data
   */
  hasBufferedData(tableName) {
    const colBuf = this._columnarBuffer.get(tableName);
    return colBuf && (colBuf.__length || 0) > 0;
  }
  async compact() {
    for (const [tableName, table] of this.tables) {
      const allRows = await this._readAllRows(tableName);
      for (const fragKey of table.fragments) {
        await opfsStorage.delete(fragKey);
      }
      table.fragments = [];
      table.deletionVector = [];
      table.rowCount = 0;
      table.nextRowId = 0;
      if (allRows.length > 0) {
        const cleanRows = allRows.map(({ __rowId, ...rest }) => rest);
        await this.insert(tableName, cleanRows);
      }
    }
    return { success: true, compacted: this.tables.size };
  }
  // Start a scan stream
  async scanStart(tableName, options = {}) {
    if (!this.tables.has(tableName)) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const streamId = nextScanId++;
    const table = this.tables.get(tableName);
    const deletedSet = new Set(table.deletionVector);
    const allRows = [];
    for (const fragKey of table.fragments) {
      const fragData = await opfsStorage.load(fragKey);
      if (fragData) {
        const rows = this._parseFragment(fragData, table.schema);
        for (const row of rows) {
          if (!deletedSet.has(row.__rowId)) {
            allRows.push(row);
          }
        }
      }
    }
    const buffer = this._writeBuffer.get(tableName);
    if (buffer) {
      for (const row of buffer) {
        if (!deletedSet.has(row.__rowId)) {
          allRows.push(row);
        }
      }
    }
    scanStreams.set(streamId, {
      rows: allRows,
      index: 0,
      batchSize: options.batchSize || 1e4,
      columns: options.columns
    });
    return streamId;
  }
  // Get next batch from scan stream
  scanNext(streamId) {
    const stream = scanStreams.get(streamId);
    if (!stream) {
      return { batch: [], done: true };
    }
    const batch = [];
    const end = Math.min(stream.index + stream.batchSize, stream.rows.length);
    for (let i = stream.index; i < end; i++) {
      const row = stream.rows[i];
      let projectedRow;
      if (stream.columns && stream.columns.length > 0 && stream.columns[0] !== "*") {
        projectedRow = {};
        for (const col of stream.columns) {
          projectedRow[col] = row[col];
        }
      } else {
        const { __rowId, ...rest } = row;
        projectedRow = rest;
      }
      batch.push(projectedRow);
    }
    stream.index = end;
    const done = stream.index >= stream.rows.length;
    if (done) {
      scanStreams.delete(streamId);
    }
    return { batch, done };
  }
};

// src/worker/worker-vault.js
var WorkerVault = class {
  constructor() {
    this._root = null;
    this._ready = false;
    this._kv = {};
    this._encryptionKeyId = null;
    this._db = null;
  }
  async open(encryptionConfig = null) {
    if (this._ready) return this;
    if (encryptionConfig) {
      const { keyId, keyBytes } = encryptionConfig;
      if (!encryptionKeys.has(keyId)) {
        const cryptoKey = await importEncryptionKey(keyBytes);
        encryptionKeys.set(keyId, cryptoKey);
      }
      this._encryptionKeyId = keyId;
    }
    try {
      const opfsRoot2 = await navigator.storage.getDirectory();
      this._root = await opfsRoot2.getDirectoryHandle("vault", { create: true });
      await this._loadKV();
      this._db = new WorkerDatabase("vault");
      await this._db.open();
      this._ready = true;
    } catch (e) {
      console.error("[WorkerVault] Failed to open OPFS:", e);
      throw e;
    }
    return this;
  }
  _getCryptoKey() {
    return this._encryptionKeyId ? encryptionKeys.get(this._encryptionKeyId) : null;
  }
  async _loadKV() {
    try {
      const cryptoKey = this._getCryptoKey();
      const filename = cryptoKey ? "_vault.json.enc" : "_vault.json";
      const fileHandle = await this._root.getFileHandle(filename);
      const file = await fileHandle.getFile();
      if (cryptoKey) {
        const buffer = await file.arrayBuffer();
        this._kv = await decryptData(new Uint8Array(buffer), cryptoKey);
      } else {
        const text = await file.text();
        this._kv = JSON.parse(text);
      }
    } catch (e) {
      if (e.name === "NotFoundError") {
        this._kv = {};
      } else {
        throw e;
      }
    }
  }
  async _saveKV() {
    const cryptoKey = this._getCryptoKey();
    const filename = cryptoKey ? "_vault.json.enc" : "_vault.json";
    const fileHandle = await this._root.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    if (cryptoKey) {
      const encrypted = await encryptData(this._kv, cryptoKey);
      await writable.write(encrypted);
    } else {
      await writable.write(JSON.stringify(this._kv));
    }
    await writable.close();
  }
  async get(key) {
    return this._kv[key];
  }
  async set(key, value) {
    this._kv[key] = value;
    await this._saveKV();
  }
  async delete(key) {
    delete this._kv[key];
    await this._saveKV();
  }
  async keys() {
    return Object.keys(this._kv);
  }
  async has(key) {
    return key in this._kv;
  }
};

// src/worker/wasm-sql-bridge.js
var WasmSqlExecutor = class {
  constructor() {
    this._registered = /* @__PURE__ */ new Map();
  }
  /**
   * Parse SQL in WASM to identify required table names
   * @param {string} sql
   * @returns {string[]}
   */
  getTableNames(sql) {
    const wasm2 = getWasm();
    if (!wasm2) throw new Error("WASM not loaded");
    const memory = getWasmMemory();
    const sqlBytes = new TextEncoder().encode(sql);
    const sqlPtr = wasm2.alloc(sqlBytes.length);
    new Uint8Array(memory.buffer, sqlPtr, sqlBytes.length).set(sqlBytes);
    const namesPtr = wasm2.getTableNames(sqlPtr, sqlBytes.length);
    if (namesPtr === 0) return [];
    const view = new Uint8Array(memory.buffer, namesPtr);
    let len = 0;
    while (view[len] !== 0 && len < 1024) len++;
    const namesStr = new TextDecoder().decode(view.subarray(0, len));
    return namesStr ? namesStr.split(",").filter((n) => n) : [];
  }
  /**
   * Check if table exists in WASM
   * @param {string} tableName
   * @returns {boolean}
   */
  hasTable(tableName) {
    const wasm2 = getWasm();
    if (!wasm2) return false;
    const memory = getWasmMemory();
    const bytes = new TextEncoder().encode(tableName);
    const ptr = wasm2.alloc(bytes.length);
    new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
    const res = wasm2.hasTable(ptr, bytes.length);
    return res === 1;
  }
  /**
   * Register table data in WASM for SQL execution
   * @param {string} tableName
   * @param {Object} columns - Map of column name to typed array
   * @param {number} rowCount
   * @param {string} version - Table version string to detect changes
   */
  registerTable(tableName, columns, rowCount, version = "") {
    const wasm2 = getWasm();
    if (!wasm2) throw new Error("WASM not loaded");
    const existing = this._registered.get(tableName);
    if (existing && existing.version === version) {
      return;
    }
    if (existing) {
      const nameBytes = new TextEncoder().encode(tableName);
      wasm2.clearTable(nameBytes, nameBytes.length);
    }
    const memory = getWasmMemory();
    const tableNameBytes = new TextEncoder().encode(tableName);
    const tableNamePtr = wasm2.alloc(tableNameBytes.length);
    new Uint8Array(memory.buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);
    const registeredCols = /* @__PURE__ */ new Set();
    for (const [colName, data] of Object.entries(columns)) {
      if (colName.startsWith("__")) continue;
      const colNameBytes = new TextEncoder().encode(colName);
      const colNamePtr = wasm2.alloc(colNameBytes.length);
      new Uint8Array(memory.buffer, colNamePtr, colNameBytes.length).set(colNameBytes);
      if (data instanceof Float64Array) {
        const dataPtr = wasm2.allocFloat64Buffer(data.length);
        new Float64Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.registerTableFloat64(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          data.length
        );
        registeredCols.add(colName);
      } else if (data instanceof BigInt64Array) {
        const dataPtr = wasm2.allocInt64Buffer(data.length);
        new BigInt64Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.registerTableInt64(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          data.length
        );
        registeredCols.add(colName);
      } else if (data instanceof Int32Array) {
        const f64Data = new Float64Array(data.length);
        for (let i = 0; i < data.length; i++) f64Data[i] = data[i];
        const dataPtr = wasm2.allocFloat64Buffer(f64Data.length);
        new Float64Array(memory.buffer, dataPtr, f64Data.length).set(f64Data);
        wasm2.registerTableFloat64(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          f64Data.length
        );
        registeredCols.add(colName);
      } else if (Array.isArray(data)) {
        const offsets = new Uint32Array(data.length);
        const lengths = new Uint32Array(data.length);
        let totalLen = 0;
        for (let i = 0; i < data.length; i++) {
          const str = String(data[i] || "");
          lengths[i] = str.length;
          offsets[i] = totalLen;
          totalLen += str.length;
        }
        const stringData = new Uint8Array(totalLen);
        let offset = 0;
        for (let i = 0; i < data.length; i++) {
          const str = String(data[i] || "");
          const bytes = new TextEncoder().encode(str);
          stringData.set(bytes, offset);
          offset += bytes.length;
        }
        const offsetsPtr = wasm2.alloc(offsets.byteLength);
        new Uint32Array(memory.buffer, offsetsPtr, offsets.length).set(offsets);
        const lengthsPtr = wasm2.alloc(lengths.byteLength);
        new Uint32Array(memory.buffer, lengthsPtr, lengths.length).set(lengths);
        const dataPtr = wasm2.alloc(stringData.length);
        new Uint8Array(memory.buffer, dataPtr, stringData.length).set(stringData);
        wasm2.registerTableString(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          offsetsPtr,
          lengthsPtr,
          dataPtr,
          totalLen,
          data.length
        );
        registeredCols.add(colName);
      }
    }
    this._registered.set(tableName, { version, columns: registeredCols, rowCount });
  }
  /**
   * Register table from OPFS file paths
   * @param {string} tableName
   * @param {string[]} filePaths - Array of OPFS paths
   * @param {string} version - Table version string
   */
  registerTableFromFiles(tableName, filePaths, version = "") {
    const wasm2 = getWasm();
    if (!wasm2) throw new Error("WASM not loaded");
    const existing = this._registered.get(tableName);
    if (existing && existing.version === version) {
      return;
    }
    if (existing) {
      const nameBytes = new TextEncoder().encode(tableName);
      wasm2.clearTable(nameBytes, nameBytes.length);
    }
    const encoder = new TextEncoder();
    const tableNameBytes = encoder.encode(tableName);
    const tableNamePtr = wasm2.alloc(tableNameBytes.length);
    new Uint8Array(getWasmMemory().buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);
    for (const path of filePaths) {
      const pathBytes = encoder.encode(path);
      const pathPtr = wasm2.alloc(pathBytes.length);
      new Uint8Array(getWasmMemory().buffer, pathPtr, pathBytes.length).set(pathBytes);
      const result = wasm2.registerTableFromOPFS(
        tableNamePtr,
        tableNameBytes.length,
        pathPtr,
        pathBytes.length
      );
      if (result !== 0) {
        console.warn(`Failed to register fragment ${path} for table ${tableName}: error ${result}`);
      }
    }
    this._registered.set(tableName, { version, type: "files" });
  }
  /**
   * Append in-memory batch to existing registered table (Hybrid Scan)
   * @param {string} tableName
   * @param {Object} columns - Map of column name to typed array
   * @param {number} rowCount
   */
  appendTableMemory(tableName, columns, rowCount) {
    const wasm2 = getWasm();
    if (!wasm2) throw new Error("WASM not loaded");
    const memory = getWasmMemory();
    const tableNameBytes = new TextEncoder().encode(tableName);
    const tableNamePtr = wasm2.alloc(tableNameBytes.length);
    new Uint8Array(memory.buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);
    for (const [colName, data] of Object.entries(columns)) {
      if (colName.startsWith("__")) continue;
      const colNameBytes = new TextEncoder().encode(colName);
      const colNamePtr = wasm2.alloc(colNameBytes.length);
      new Uint8Array(memory.buffer, colNamePtr, colNameBytes.length).set(colNameBytes);
      if (data instanceof Float64Array) {
        const dataPtr = wasm2.allocFloat64Buffer(data.length);
        new Float64Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.appendTableMemory(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          4,
          rowCount
        );
      } else if (data instanceof BigInt64Array) {
        const dataPtr = wasm2.allocInt64Buffer(data.length);
        new BigInt64Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.appendTableMemory(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          2,
          rowCount
        );
      } else if (data instanceof Int32Array) {
        const dataPtr = wasm2.alloc(data.byteLength);
        new Int32Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.appendTableMemory(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          1,
          rowCount
        );
      } else if (data instanceof Float32Array) {
        const dataPtr = wasm2.alloc(data.byteLength);
        new Float32Array(memory.buffer, dataPtr, data.length).set(data);
        wasm2.appendTableMemory(
          tableNamePtr,
          tableNameBytes.length,
          colNamePtr,
          colNameBytes.length,
          dataPtr,
          3,
          rowCount
        );
      }
    }
  }
  /**
   * Execute SQL and return result as columnar data
   * @param {string} sql - SQL query string
   * @returns {Object} - { columns: string[], rowCount: number, data: Object }
   */
  execute(sql) {
    const wasm2 = getWasm();
    if (!wasm2) throw new Error("WASM not loaded");
    const memory = getWasmMemory();
    const sqlInputPtr = wasm2.getSqlInputBuffer();
    const sqlInputSize = wasm2.getSqlInputBufferSize();
    const sqlBytes = new TextEncoder().encode(sql);
    if (sqlBytes.length > sqlInputSize) {
      throw new Error(`SQL too long: ${sqlBytes.length} > ${sqlInputSize}`);
    }
    new Uint8Array(memory.buffer, sqlInputPtr, sqlBytes.length).set(sqlBytes);
    wasm2.setSqlInputLength(sqlBytes.length);
    const resultPtr = wasm2.executeSql();
    if (resultPtr === 0) {
      throw new Error("WASM SQL execution failed");
    }
    const resultSize = wasm2.getResultSize();
    const result = this._parseResult(memory.buffer, resultPtr, resultSize);
    wasm2.resetResult();
    return result;
  }
  /**
   * Parse WASM result buffer (Lance File/Fragment format)
   */
  _parseResult(buffer, ptr, size) {
    const view = new DataView(buffer, ptr, size);
    const decoder = new TextDecoder();
    if (size < 40) throw new Error("Result too small for Lance footer");
    const footerOffset = size - 40;
    const magicVals = [
      view.getUint8(footerOffset + 36),
      view.getUint8(footerOffset + 37),
      view.getUint8(footerOffset + 38),
      view.getUint8(footerOffset + 39)
    ];
    const magic = String.fromCharCode(...magicVals);
    if (magic !== "LANC") {
      throw new Error("Invalid Lance Magic: " + magic);
    }
    const colMetaOffsetsStart = Number(view.getBigUint64(footerOffset + 8, true));
    const numCols = view.getUint32(footerOffset + 28, true);
    const columns = [];
    const colData = {};
    let resultRowCount = 0;
    for (let i = 0; i < numCols; i++) {
      const offsetPos = colMetaOffsetsStart + i * 8;
      const metaPos = Number(view.getBigUint64(offsetPos, true));
      let localOffset = metaPos;
      view.getUint8(localOffset++);
      const [nameLen, lenBytes] = this._readVarint(view, localOffset);
      localOffset += lenBytes;
      const nameBytes = new Uint8Array(buffer, ptr + localOffset, nameLen);
      const colName = decoder.decode(nameBytes);
      localOffset += nameLen;
      columns.push(colName);
      view.getUint8(localOffset++);
      const [typeLen, typeLenBytes] = this._readVarint(view, localOffset);
      localOffset += typeLenBytes;
      const typeBytes = new Uint8Array(buffer, ptr + localOffset, typeLen);
      const typeStr = decoder.decode(typeBytes);
      localOffset += typeLen;
      view.getUint8(localOffset++);
      const [nullable, nullBytes] = this._readVarint(view, localOffset);
      localOffset += nullBytes;
      view.getUint8(localOffset++);
      const dataOffset = Number(view.getBigUint64(localOffset, true));
      localOffset += 8;
      view.getUint8(localOffset++);
      const [rowCount, rowBytes] = this._readVarint(view, localOffset);
      localOffset += rowBytes;
      resultRowCount = rowCount;
      view.getUint8(localOffset++);
      const [dataSize, sizeBytes] = this._readVarint(view, localOffset);
      localOffset += sizeBytes;
      const absDataOffset = ptr + dataOffset;
      if (typeStr === "float64" || typeStr === "int64" || typeStr === "int32" || typeStr === "float32") {
        if (typeStr === "float64") {
          colData[colName] = new Float64Array(buffer, absDataOffset, rowCount).slice();
        } else if (typeStr === "int64") {
          const src = new BigInt64Array(buffer, absDataOffset, rowCount);
          const dst = new Float64Array(rowCount);
          for (let j = 0; j < rowCount; j++) dst[j] = Number(src[j]);
          colData[colName] = dst;
        } else if (typeStr === "int32") {
          const src = new Int32Array(buffer, absDataOffset, rowCount);
          const dst = new Float64Array(rowCount);
          for (let j = 0; j < rowCount; j++) dst[j] = src[j];
          colData[colName] = dst;
        } else {
          const src = new Float32Array(buffer, absDataOffset, rowCount);
          const dst = new Float64Array(rowCount);
          for (let j = 0; j < rowCount; j++) dst[j] = src[j];
          colData[colName] = dst;
        }
      } else if (typeStr === "string") {
        const offsetsLen = (rowCount + 1) * 4;
        const dataBytesLen = dataSize - offsetsLen;
        const bytes = new Uint8Array(buffer, absDataOffset, dataBytesLen).slice();
        const offsets = new Uint32Array(buffer, absDataOffset + dataBytesLen, rowCount + 1).slice();
        colData[colName] = { _arrowString: true, offsets, bytes };
      }
    }
    return {
      _format: "columnar",
      columns,
      rowCount: resultRowCount,
      data: colData
    };
  }
  _readVarint(view, offset) {
    let result = 0;
    let shift = 0;
    let bytesRead = 0;
    while (true) {
      const byte = view.getUint8(offset + bytesRead);
      bytesRead++;
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return [result, bytesRead];
  }
  /**
   * Clear all registered tables
   */
  clear() {
    const wasm2 = getWasm();
    if (wasm2) {
      wasm2.clearTables();
    }
    this._registered.clear();
  }
};
var instance = null;
function getWasmSqlExecutor() {
  if (!instance) {
    instance = new WasmSqlExecutor();
  }
  return instance;
}

// src/worker/index.js
async function executeWasmSqlFull(db, sql) {
  if (!wasm) {
    await loadWasm();
    if (!wasm) throw new Error("WASM not loaded");
  }
  const executor = getWasmSqlExecutor();
  const tableNames = executor.getTableNames(sql);
  console.log(`[LanceQLWorker] executeWasmSqlFull: ${sql} -> tables: ${tableNames.join(",")}`);
  for (const tableName of tableNames) {
    const exists = executor.hasTable(tableName);
    console.log(`[LanceQLWorker] Check hasTable '${tableName}': ${exists}`);
    if (exists) continue;
    const table = db.tables.get(tableName);
    if (!table) continue;
    const colBuf = db._columnarBuffer?.get(tableName);
    const bufLen = colBuf?.__length || 0;
    const version = `${tableName}:${table.fragments?.length || 0}:${bufLen}:${table.deletionVector?.length || 0}`;
    const hasFiles = table.fragments.length > 0;
    const hasMemory = bufLen > 0;
    if (hasFiles) {
      const handles = [];
      for (const fragPath of table.fragments) {
        const handleId = await registerOPFSFile(fragPath);
        if (handleId) handles.push(handleId);
      }
      executor.registerTableFromFiles(tableName, table.fragments, version);
      if (hasMemory) {
        const columns = {};
        for (const c of table.schema) {
          const arr = colBuf[c.name];
          if (arr && ArrayBuffer.isView(arr)) {
            columns[c.name] = arr.subarray(0, bufLen);
          }
        }
        executor.appendTableMemory(tableName, columns, bufLen);
      }
    } else if (hasMemory) {
      const columnarData = await db.selectColumnar(tableName);
      if (columnarData) {
        const { columns, rowCount } = columnarData;
        executor.registerTable(tableName, columns, rowCount, version);
      }
    }
  }
  return executor.execute(sql);
}
var wasm = null;
var wasmMemory = null;
var opfsHandles = /* @__PURE__ */ new Map();
var nextHandleId = 1;
async function getOPFSDir(pathParts) {
  let current = await navigator.storage.getDirectory();
  current = await current.getDirectoryHandle("lanceql", { create: true });
  for (const part of pathParts) {
    current = await current.getDirectoryHandle(part, { create: true });
  }
  return current;
}
function createWasmImports() {
  return {
    env: {
      // Open file, returns handle ID (0 = error)
      opfs_open: (pathPtr, pathLen) => {
        try {
          const pathBytes = new Uint8Array(wasmMemory.buffer, pathPtr, pathLen);
          const path = new TextDecoder().decode(pathBytes);
          for (const [id, handle] of opfsHandles.entries()) {
            if (handle._path === path) {
              return id;
            }
          }
          console.warn("[LanceQLWorker] WASM tried to open unregistered path:", path);
          return 0;
        } catch (e) {
          console.error("[LanceQLWorker] Error:", e);
          return 0;
        }
      },
      // Read from file at offset into buffer
      opfs_read: (handle, bufPtr, bufLen, offset) => {
        const accessHandle = opfsHandles.get(handle);
        if (!accessHandle) return 0;
        try {
          const buf = new Uint8Array(wasmMemory.buffer, bufPtr, bufLen);
          return accessHandle.read(buf, { at: Number(offset) });
        } catch (e) {
          return 0;
        }
      },
      // Get file size
      opfs_size: (handle) => {
        const accessHandle = opfsHandles.get(handle);
        if (!accessHandle) return BigInt(0);
        try {
          return BigInt(accessHandle.getSize());
        } catch (e) {
          return BigInt(0);
        }
      },
      // Close file handle
      opfs_close: (handle) => {
      },
      js_log: (ptr, len) => {
        try {
          const bytes = new Uint8Array(wasmMemory.buffer, ptr, len);
          const msg = new TextDecoder().decode(bytes);
          console.log(`[LanceQLWasm] ${msg}`);
        } catch (e) {
          console.error("[LanceQLWasm] Log error:", e);
        }
      },
      __assert_fail: (msgPtr, filePtr, line, funcPtr) => {
        const decoder = new TextDecoder();
        const msg = decoder.decode(new Uint8Array(wasmMemory.buffer, msgPtr).subarray(0, 100));
        console.error(`[WASM ASSERT] ${msg} at line ${line}`);
      }
    }
  };
}
async function registerOPFSFile(path) {
  try {
    const parts = path.split("/").filter((p) => p);
    const fileName = parts.pop();
    const dir = await getOPFSDir(parts);
    const fileHandle = await dir.getFileHandle(fileName);
    const accessHandle = await fileHandle.createSyncAccessHandle();
    const handleId = nextHandleId++;
    accessHandle._path = path;
    opfsHandles.set(handleId, accessHandle);
    return handleId;
  } catch (e) {
    console.warn("[LanceQLWorker] Failed to register OPFS file:", path, e);
    return 0;
  }
}
function closeOPFSFile(handleId) {
  const accessHandle = opfsHandles.get(handleId);
  if (accessHandle) {
    try {
      accessHandle.close();
    } catch (e) {
    }
    opfsHandles.delete(handleId);
  }
}
async function loadWasm() {
  if (wasm) return wasm;
  try {
    const response = await fetch(new URL("./lanceql.wasm", import.meta.url));
    const bytes = await response.arrayBuffer();
    const imports = createWasmImports();
    const module = await WebAssembly.instantiate(bytes, imports);
    wasm = module.instance.exports;
    wasmMemory = wasm.memory;
    console.log("[LanceQLWorker] WASM loaded with OPFS support");
    return wasm;
  } catch (e) {
    console.warn("[LanceQLWorker] WASM not available:", e.message);
    return null;
  }
}
function getWasm() {
  return wasm;
}
function getWasmMemory() {
  return wasmMemory;
}
var wasmBufferPtr = 0;
var wasmBufferSize = 0;
var MIN_BUFFER_SIZE = 1024 * 1024;
function getWasmBuffer(size) {
  if (!wasm) return 0;
  if (size <= wasmBufferSize && wasmBufferPtr !== 0) {
    return wasmBufferPtr;
  }
  const newSize = Math.max(size, MIN_BUFFER_SIZE);
  const ptr = wasm.alloc(newSize);
  if (ptr) {
    wasmBufferPtr = ptr;
    wasmBufferSize = newSize;
  }
  return ptr;
}
async function loadFragmentToWasm(fragPath) {
  const w = await loadWasm();
  if (!w) return 0;
  try {
    const parts = fragPath.split("/").filter((p) => p);
    const fileName = parts.pop();
    const dir = await getOPFSDir(parts);
    const fileHandle = await dir.getFileHandle(fileName);
    const accessHandle = await fileHandle.createSyncAccessHandle();
    const size = accessHandle.getSize();
    const ptr = getWasmBuffer(size);
    if (!ptr) {
      accessHandle.close();
      return 0;
    }
    const wasmBuf = new Uint8Array(wasmMemory.buffer, ptr, size);
    const bytesRead = accessHandle.read(wasmBuf, { at: 0 });
    accessHandle.close();
    if (bytesRead !== size) return 0;
    return w.openFile(ptr, size);
  } catch (e) {
    console.warn("[LanceQLWorker] Failed to load fragment:", fragPath, e);
    return 0;
  }
}
async function getFileHandle(root, path) {
  const parts = path.split("/").filter((p) => p);
  let currentDir = root;
  for (let i = 0; i < parts.length - 1; i++) {
    currentDir = await currentDir.getDirectoryHandle(parts[i], { create: false });
  }
  return await currentDir.getFileHandle(parts[parts.length - 1], { create: false });
}
var opfsRoot = null;
async function loadFragmentCached(fragPath) {
  let loaded = bufferPool.get(fragPath);
  if (loaded) return loaded;
  if (!opfsRoot) {
    opfsRoot = await navigator.storage.getDirectory();
  }
  const handle = await getFileHandle(opfsRoot, fragPath);
  const file = await handle.getFile();
  const buffer = await file.arrayBuffer();
  loaded = new Uint8Array(buffer);
  if (loaded) {
    bufferPool.set(fragPath, loaded, loaded.byteLength);
  }
  return loaded;
}
async function wasmAggregate(fragPath, colIdx, func) {
  const loaded = await loadFragmentCached(fragPath);
  if (!loaded) return null;
  const w = wasm;
  switch (func) {
    case "sum":
      return w.opfsSumFloat64Column(colIdx);
    case "min":
      return w.opfsMinFloat64Column(colIdx);
    case "max":
      return w.opfsMaxFloat64Column(colIdx);
    case "avg":
      return w.opfsAvgFloat64Column(colIdx);
    case "count":
      return Number(w.opfsCountRows());
    default:
      return null;
  }
}
loadWasm();
var bufferPool = new BufferPool();
var stores = /* @__PURE__ */ new Map();
var databases = /* @__PURE__ */ new Map();
var vaultInstance = null;
var ports = /* @__PURE__ */ new Set();
var sharedBuffer = null;
var sharedOffset = 0;
var SHARED_THRESHOLD = 1024;
var cursors = /* @__PURE__ */ new Map();
var nextCursorId = 1;
async function getVault(encryptionConfig = null) {
  if (!vaultInstance) {
    vaultInstance = new WorkerVault();
  }
  await vaultInstance.open(encryptionConfig);
  return vaultInstance;
}
async function getStore(name, options = {}, encryptionConfig = null) {
  const encKeyId = encryptionConfig?.keyId || "none";
  const key = `${name}:${encKeyId}:${JSON.stringify(options)}`;
  if (!stores.has(key)) {
    const store = new WorkerStore(name, options);
    await store.open(encryptionConfig);
    stores.set(key, store);
  }
  return stores.get(key);
}
async function getDatabase(name) {
  if (!databases.has(name)) {
    const db = new WorkerDatabase(name, bufferPool);
    await db.open();
    databases.set(name, db);
  }
  return databases.get(name);
}
function sendResponse(port, id, result) {
  if (result && result._format === "wasm_binary") {
    port.postMessage({
      id,
      result: {
        _format: "wasm_binary",
        buffer: result.buffer,
        columns: result.columns,
        rowCount: result.rowCount,
        schema: result.schema
      }
    }, [result.buffer]);
    return;
  }
  if (result && result._format === "columnar" && result.data) {
    const colNames = result.columns;
    const rowCount = result.rowCount;
    if (rowCount < 1e5) {
      const transferables2 = [];
      const serializedData = {};
      const usedBuffers = /* @__PURE__ */ new Set();
      for (const name of colNames) {
        const arr = result.data[name];
        if (ArrayBuffer.isView(arr)) {
          const isView = arr.byteOffset !== 0 || arr.byteLength < arr.buffer.byteLength;
          const bufferAlreadyUsed = usedBuffers.has(arr.buffer);
          if (isView || bufferAlreadyUsed) {
            const copy = new arr.constructor(arr);
            serializedData[name] = copy;
            transferables2.push(copy.buffer);
          } else {
            serializedData[name] = arr;
            transferables2.push(arr.buffer);
            usedBuffers.add(arr.buffer);
          }
        } else if (arr && arr._arrowString) {
          serializedData[name] = arr;
          if (arr.offsets && arr.offsets.buffer && !usedBuffers.has(arr.offsets.buffer)) {
            transferables2.push(arr.offsets.buffer);
            usedBuffers.add(arr.offsets.buffer);
          }
          if (arr.bytes && arr.bytes.buffer && !usedBuffers.has(arr.bytes.buffer)) {
            transferables2.push(arr.bytes.buffer);
            usedBuffers.add(arr.bytes.buffer);
          }
        } else {
          serializedData[name] = arr;
        }
      }
      port.postMessage({
        id,
        result: { _format: "columnar", columns: colNames, rowCount, data: serializedData }
      }, transferables2);
      return;
    }
    const typedCols = [];
    const stringCols = [];
    let numericBytes = 0;
    for (const name of colNames) {
      const arr = result.data[name];
      if (ArrayBuffer.isView(arr)) {
        typedCols.push({ name, arr });
        numericBytes += arr.byteLength;
      } else if (Array.isArray(arr)) {
        stringCols.push({ name, arr });
      }
    }
    const packedBuffer = numericBytes > 0 ? new ArrayBuffer(numericBytes) : null;
    const colOffsets = {};
    let offset = 0;
    if (packedBuffer) {
      const packedView = new Uint8Array(packedBuffer);
      for (const { name, arr } of typedCols) {
        const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
        packedView.set(bytes, offset);
        colOffsets[name] = { offset, length: arr.length, type: arr.constructor.name };
        offset += arr.byteLength;
      }
    }
    const stringData = {};
    for (const { name, arr } of stringCols) {
      stringData[name] = arr;
    }
    const transferables = [];
    if (packedBuffer) transferables.push(packedBuffer);
    port.postMessage({
      id,
      result: {
        _format: "packed",
        columns: colNames,
        rowCount,
        packedBuffer,
        colOffsets,
        stringData
      }
    }, transferables);
    return;
  }
  if (sharedBuffer && result !== void 0) {
    const json = JSON.stringify(result);
    if (json.length > SHARED_THRESHOLD) {
      const bytes = E.encode(json);
      if (sharedOffset + bytes.length <= sharedBuffer.byteLength) {
        const view = new Uint8Array(sharedBuffer, sharedOffset, bytes.length);
        view.set(bytes);
        port.postMessage({
          id,
          sharedOffset,
          sharedLength: bytes.length
        });
        sharedOffset += bytes.length;
        if (sharedOffset > sharedBuffer.byteLength / 2) {
          sharedOffset = 0;
        }
        return;
      }
    }
  }
  port.postMessage({ id, result });
}
async function handleMessage(port, data) {
  if (data.type === "initSharedBuffer") {
    sharedBuffer = data.buffer;
    sharedOffset = 0;
    console.log("[LanceQLWorker] SharedArrayBuffer initialized:", sharedBuffer.byteLength, "bytes");
    return;
  }
  const { id, method, args } = data;
  try {
    let result;
    if (method === "ping") {
      result = "pong";
    } else if (method === "open") {
      await getStore(args.name, args.options, args.encryption);
      result = true;
    } else if (method === "get") {
      result = await (await getStore(args.name)).get(args.key);
    } else if (method === "set") {
      await (await getStore(args.name)).set(args.key, args.value);
      result = true;
    } else if (method === "delete") {
      await (await getStore(args.name)).delete(args.key);
      result = true;
    } else if (method === "keys") {
      result = await (await getStore(args.name)).keys();
    } else if (method === "clear") {
      await (await getStore(args.name)).clear();
      result = true;
    } else if (method === "filter") {
      result = await (await getStore(args.name)).filter(args.key, args.query);
    } else if (method === "find") {
      result = await (await getStore(args.name)).find(args.key, args.query);
    } else if (method === "search") {
      result = await (await getStore(args.name)).search(args.key, args.text, args.limit);
    } else if (method === "count") {
      result = await (await getStore(args.name)).count(args.key, args.query);
    } else if (method === "enableSemanticSearch") {
      result = await (await getStore(args.name)).enableSemanticSearch(args.options);
    } else if (method === "disableSemanticSearch") {
      (await getStore(args.name)).disableSemanticSearch();
      result = true;
    } else if (method === "hasSemanticSearch") {
      result = (await getStore(args.name)).hasSemanticSearch();
    } else if (method === "db:open") {
      console.log(`[LanceQLWorker] db:open ${args.name}`);
      await getDatabase(args.name);
      result = true;
    } else if (method === "db:createTable") {
      console.log(`[LanceQLWorker] db:createTable ${args.tableName}`);
      result = await (await getDatabase(args.db)).createTable(args.tableName, args.columns, args.ifNotExists);
      console.log(`[LanceQLWorker] db:createTable ${args.tableName} done`);
    } else if (method === "db:dropTable") {
      console.log(`[LanceQLWorker] db:dropTable ${args.tableName}`);
      const db = await getDatabase(args.db);
      result = await db.dropTable(args.tableName, args.ifExists);
      const nameBytes = E.encode(args.tableName);
      getWasmSqlExecutor().clearTable(nameBytes, nameBytes.length);
    } else if (method === "db:insert") {
      console.log(`[LanceQLWorker] db:insert into ${args.tableName}, rows: ${args.rows?.length}`);
      result = await (await getDatabase(args.db)).insert(args.tableName, args.rows);
      console.log(`[LanceQLWorker] db:insert done`);
    } else if (method === "db:delete") {
      const db = await getDatabase(args.db);
      const predicate = args.where ? (row) => evalWhere(args.where, row) : () => true;
      result = await db.delete(args.tableName, predicate);
    } else if (method === "db:update") {
      const db = await getDatabase(args.db);
      const predicate = args.where ? (row) => evalWhere(args.where, row) : () => true;
      result = await db.update(args.tableName, args.updates, predicate);
    } else if (method === "db:select") {
      const db = await getDatabase(args.db);
      const options = { ...args.options };
      if (args.where) {
        options.where = (row) => evalWhere(args.where, row);
      }
      result = await db.select(args.tableName, options);
    } else if (method === "db:exec") {
      const db = await getDatabase(args.db);
      const executor = getWasmSqlExecutor();
      const tableNames = executor.getTableNames(args.sql);
      const primaryTable = tableNames[0] ? db.tables.get(tableNames[0]) : null;
      const rowCount = primaryTable ? primaryTable.rowCount : 0;
      result = await executeWasmSqlFull(db, args.sql);
      if (result && result._format === "columnar" && result.rowCount >= 1e5) {
        const cursorId = nextCursorId++;
        cursors.set(cursorId, result);
        result = {
          _format: "cursor",
          cursorId,
          columns: result.columns,
          rowCount: result.rowCount
        };
      }
    } else if (method === "cursor:fetch") {
      const cursor = cursors.get(args.cursorId);
      if (!cursor) throw new Error("Cursor not found");
      result = cursor;
      cursors.delete(args.cursorId);
    } else if (method === "db:flush") {
      console.log(`[LanceQLWorker] db:flush ${args.db}`);
      await (await getDatabase(args.db)).flush();
      console.log(`[LanceQLWorker] db:flush ${args.db} done`);
      result = true;
    } else if (method === "db:compact") {
      result = await (await getDatabase(args.db)).compact();
    } else if (method === "db:listTables") {
      result = (await getDatabase(args.db)).listTables();
    } else if (method === "db:getTable") {
      result = (await getDatabase(args.db)).getTable(args.tableName);
    } else if (method === "db:scanStart") {
      result = await (await getDatabase(args.db)).scanStart(args.tableName, args.options);
    } else if (method === "db:scanNext") {
      const db = await getDatabase(args.db);
      result = db.scanNext(args.streamId);
    } else if (method === "vault:open") {
      await getVault(args.encryption);
      result = true;
    } else if (method === "vault:get") {
      result = await (await getVault()).get(args.key);
    } else if (method === "vault:set") {
      await (await getVault()).set(args.key, args.value);
      result = true;
    } else if (method === "vault:delete") {
      await (await getVault()).delete(args.key);
      result = true;
    } else if (method === "vault:keys") {
      result = await (await getVault()).keys();
    } else if (method === "vault:has") {
      result = await (await getVault()).has(args.key);
    } else if (method === "vault:exec") {
      const vault = await getVault();
      const executor = getWasmSqlExecutor();
      const tableNames = executor.getTableNames(args.sql);
      const primaryTable = tableNames[0] ? vault._db.tables.get(tableNames[0]) : null;
      const rowCount = primaryTable ? primaryTable.rowCount : 0;
      result = await executeWasmSqlFull(vault._db, args.sql);
      if (result && result._format === "columnar" && result.rowCount >= 1e5) {
        const cursorId = nextCursorId++;
        cursors.set(cursorId, result);
        result = {
          _format: "cursor",
          cursorId,
          columns: result.columns,
          rowCount: result.rowCount
        };
      }
    } else {
      throw new Error(`Unknown method: ${method}`);
    }
    sendResponse(port, id, result);
  } catch (error) {
    port.postMessage({ id, error: error.stack || error.message });
  }
}
var isSharedWorker = typeof SharedWorkerGlobalScope !== "undefined" && self instanceof SharedWorkerGlobalScope;
if (isSharedWorker) {
  self.onconnect = (event) => {
    const port = event.ports[0];
    ports.add(port);
    port.onmessage = (e) => {
      handleMessage(port, e.data);
    };
    port.onmessageerror = (e) => {
      console.error("[LanceQLWorker] Message error:", e);
    };
    loadWasm().then(() => {
      port.postMessage({ type: "ready" });
    }).catch((err) => {
      console.error("[LanceQLWorker] Failed to load WASM:", err);
      port.postMessage({ type: "ready", error: "WASM load failed" });
    });
    port.start();
    console.log("[LanceQLWorker] New connection, total ports:", ports.size);
  };
} else {
  self.onmessage = (e) => {
    handleMessage(self, e.data);
  };
  loadWasm().then(() => {
    self.postMessage({ type: "ready" });
  }).catch((err) => {
    console.error("[LanceQLWorker] Failed to load WASM:", err);
    self.postMessage({ type: "ready", error: "WASM load failed" });
  });
}
console.log("[LanceQLWorker] Initialized");
export {
  closeOPFSFile,
  getWasm,
  getWasmMemory,
  loadFragmentToWasm,
  registerOPFSFile,
  wasmAggregate
};
//# sourceMappingURL=lanceql-worker.js.map
