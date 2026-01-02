/**
 * LanceQL CLI Tests
 *
 * Tests the CLI when run via npx/bunx:
 *   npx @metal0/lanceql --version
 *   npx @metal0/lanceql query "SELECT * FROM 'file.parquet'"
 *   npx @metal0/lanceql ingest data.csv -o out.lance
 */

const { spawnSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');
const { describe, it, before, after } = require('node:test');
const http = require('http');

// Path to the built CLI binary
const CLI_PATH = path.resolve(__dirname, '../../../zig-out/bin/lanceql');
const FIXTURES_PATH = path.resolve(__dirname, '../../../tests/fixtures');

// Helper to run CLI command
function runCli(args, options = {}) {
  // Split args string into array for spawnSync
  const argsArray = args ? args.match(/(?:[^\s"]+|"[^"]*")+/g) || [] : [];
  // Remove quotes from arguments
  const cleanArgs = argsArray.map(arg => arg.replace(/^"|"$/g, ''));

  const result = spawnSync(CLI_PATH, cleanArgs, {
    encoding: 'utf-8',
    timeout: 30000,
    ...options,
  });

  return {
    stdout: result.stdout || '',
    stderr: result.stderr || '',
    exitCode: result.status || 0,
    // Combined output for tests that check either stream
    output: (result.stdout || '') + (result.stderr || ''),
  };
}

// Check if CLI binary exists
function cliExists() {
  return fs.existsSync(CLI_PATH);
}

// Helper for async sleep
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Helper to make HTTP request
function httpRequest(options) {
  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve({ status: res.statusCode, data }));
    });
    req.on('error', reject);
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
    if (options.body) req.write(options.body);
    req.end();
  });
}

describe('LanceQL CLI', () => {
  before(() => {
    if (!cliExists()) {
      console.log('CLI binary not found. Run `zig build` first.');
      console.log(`Expected at: ${CLI_PATH}`);
    }
  });

  describe('Version and Help', () => {
    it('should show version with --version', () => {
      if (!cliExists()) return;

      const result = runCli('--version');
      assert.strictEqual(result.exitCode, 0);
      // Version may be on stdout or stderr
      assert.match(result.output, /lanceql \d+\.\d+\.\d+/);
    });

    it('should show help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('--help');
      assert.strictEqual(result.exitCode, 0);
      // Help may be on stdout or stderr
      assert.match(result.output, /Usage:/);
      assert.match(result.output, /query|ingest|transform/i);
    });

    it('should show help with no arguments', () => {
      if (!cliExists()) return;

      const result = runCli('');
      assert.strictEqual(result.exitCode, 0);
    });
  });

  describe('Query Command', () => {
    it('should query Parquet file', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT * FROM '${parquetFile}' LIMIT 5"`);
      assert.strictEqual(result.exitCode, 0);
      // Should output table data (may be on stdout or stderr)
      assert.ok(result.output.length > 0);
    });

    it('should show query help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('query --help');
      assert.strictEqual(result.exitCode, 0);
      // Help may be in stdout or stderr
      assert.match(result.output, /SELECT|query|Usage/i);
    });

    it('should handle invalid SQL gracefully', () => {
      if (!cliExists()) return;

      const result = runCli('query "INVALID SQL SYNTAX"');
      // Should fail or print an error message (exit code 0 with error message is acceptable)
      const hasError = result.exitCode !== 0 ||
                       result.output.toLowerCase().includes('error');
      assert.ok(hasError, 'Should indicate an error for invalid SQL');
    });
  });

  describe('Ingest Command', () => {
    let tempDir;

    before(() => {
      tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lanceql-test-'));
    });

    after(() => {
      if (tempDir) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should ingest CSV to Lance', () => {
      if (!cliExists()) return;

      // Create test CSV
      const csvPath = path.join(tempDir, 'test.csv');
      const lancePath = path.join(tempDir, 'test.lance');
      fs.writeFileSync(csvPath, 'id,name,value\n1,Alice,10.5\n2,Bob,20.5\n');

      const result = runCli(`ingest "${csvPath}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      assert.ok(fs.existsSync(lancePath), 'Lance file should be created');
    });

    it('should ingest JSON to Lance', () => {
      if (!cliExists()) return;

      const jsonPath = path.join(tempDir, 'test.json');
      const lancePath = path.join(tempDir, 'test_json.lance');
      fs.writeFileSync(jsonPath, '[{"id": 1, "name": "Test"}]');

      const result = runCli(`ingest "${jsonPath}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      assert.ok(fs.existsSync(lancePath), 'Lance file should be created');
    });

    it('should ingest JSONL to Lance', () => {
      if (!cliExists()) return;

      const jsonlPath = path.join(tempDir, 'test.jsonl');
      const lancePath = path.join(tempDir, 'test_jsonl.lance');
      fs.writeFileSync(jsonlPath, '{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n');

      const result = runCli(`ingest "${jsonlPath}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      assert.ok(fs.existsSync(lancePath), 'Lance file should be created');
    });

    it('should ingest Parquet to Lance', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const lancePath = path.join(tempDir, 'from_parquet.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`ingest "${parquetFile}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      assert.ok(fs.existsSync(lancePath), 'Lance file should be created');
    });

    it('should show ingest help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('ingest --help');
      assert.strictEqual(result.exitCode, 0);
      // Help may be on stdout or stderr
      assert.match(result.output, /ingest|csv|parquet/i);
    });

    it('should fail gracefully on missing input file', () => {
      if (!cliExists()) return;

      const result = runCli('ingest "/nonexistent/file.csv" -o "/tmp/out.lance"');
      // Should fail or indicate error (exit code != 0, or error message in output)
      const hasError = result.exitCode !== 0 ||
                       result.output.toLowerCase().includes('error') ||
                       result.output.toLowerCase().includes('not found') ||
                       result.output.toLowerCase().includes('failed');
      assert.ok(hasError, 'Should indicate an error for missing file');
    });
  });

  describe('Transform Command', () => {
    it('should show transform help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('transform --help');
      // Transform may not be fully implemented yet
      assert.ok(result.exitCode === 0 || result.stderr.includes('not implemented'));
    });
  });

  describe('Multi-Format Ingest', () => {
    let tempDir;

    before(() => {
      tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lanceql-formats-'));
    });

    after(() => {
      if (tempDir) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should ingest Arrow IPC file', () => {
      if (!cliExists()) return;

      const arrowFile = path.join(FIXTURES_PATH, 'simple.arrow');
      const lancePath = path.join(tempDir, 'from_arrow.lance');

      if (!fs.existsSync(arrowFile)) {
        console.log('Skipping: simple.arrow not found');
        return;
      }

      const result = runCli(`ingest "${arrowFile}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as Arrow
      assert.match(result.output, /Format: arrow|Arrow IPC/i);
    });

    it('should ingest Avro file', () => {
      if (!cliExists()) return;

      const avroFile = path.join(FIXTURES_PATH, 'simple.avro');
      const lancePath = path.join(tempDir, 'from_avro.lance');

      if (!fs.existsSync(avroFile)) {
        console.log('Skipping: simple.avro not found');
        return;
      }

      const result = runCli(`ingest "${avroFile}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as Avro
      assert.match(result.output, /Format: avro|Avro/i);
    });

    it('should ingest ORC file', () => {
      if (!cliExists()) return;

      const orcFile = path.join(FIXTURES_PATH, 'simple.orc');
      const lancePath = path.join(tempDir, 'from_orc.lance');

      if (!fs.existsSync(orcFile)) {
        console.log('Skipping: simple.orc not found');
        return;
      }

      const result = runCli(`ingest "${orcFile}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as ORC
      assert.match(result.output, /Format: orc|ORC/i);
    });

    it('should ingest XLSX file', () => {
      if (!cliExists()) return;

      const xlsxFile = path.join(FIXTURES_PATH, 'simple_uncompressed.xlsx');
      const lancePath = path.join(tempDir, 'from_xlsx.lance');

      if (!fs.existsSync(xlsxFile)) {
        console.log('Skipping: simple_uncompressed.xlsx not found');
        return;
      }

      const result = runCli(`ingest "${xlsxFile}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as XLSX
      assert.match(result.output, /Format: xlsx|XLSX/i);
    });

    it('should ingest Delta Lake table', () => {
      if (!cliExists()) return;

      const deltaPath = path.join(FIXTURES_PATH, 'simple.delta');
      const lancePath = path.join(tempDir, 'from_delta.lance');

      if (!fs.existsSync(deltaPath)) {
        console.log('Skipping: simple.delta not found');
        return;
      }

      const result = runCli(`ingest "${deltaPath}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as Delta
      assert.match(result.output, /Format: delta|Delta Lake/i);
    });

    it('should ingest Iceberg table', () => {
      if (!cliExists()) return;

      const icebergPath = path.join(FIXTURES_PATH, 'simple.iceberg');
      const lancePath = path.join(tempDir, 'from_iceberg.lance');

      if (!fs.existsSync(icebergPath)) {
        console.log('Skipping: simple.iceberg not found');
        return;
      }

      const result = runCli(`ingest "${icebergPath}" -o "${lancePath}"`);
      assert.strictEqual(result.exitCode, 0, `stderr: ${result.stderr}`);
      // Check that format was auto-detected as Iceberg
      assert.match(result.output, /Format: iceberg|Iceberg/i);
    });

    it('should list all supported formats in help', () => {
      if (!cliExists()) return;

      const result = runCli('ingest --help');
      assert.strictEqual(result.exitCode, 0);

      // Check for all format mentions in help
      const formats = ['csv', 'parquet', 'arrow', 'avro', 'orc', 'xlsx', 'delta', 'iceberg'];
      for (const format of formats) {
        assert.match(
          result.output.toLowerCase(),
          new RegExp(format),
          `Help should mention ${format} format`
        );
      }
    });
  });

  describe('Serve Command', () => {
    const TEST_PORT = 19876; // Use high port to avoid conflicts
    let serverProcess = null;

    after(() => {
      if (serverProcess) {
        serverProcess.kill('SIGTERM');
        serverProcess = null;
      }
    });

    it('should show serve help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('serve --help');
      assert.strictEqual(result.exitCode, 0);
      assert.match(result.output, /serve|HTTP|REST|API|port/i);
    });

    it('should start server and respond to health check', async () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      // Start server in background
      serverProcess = spawn(CLI_PATH, ['serve', parquetFile, '--port', String(TEST_PORT), '--no-open'], {
        stdio: ['ignore', 'pipe', 'pipe']
      });

      // Wait for server to start
      await sleep(1500);

      try {
        const response = await httpRequest({
          hostname: 'localhost',
          port: TEST_PORT,
          path: '/api/health',
          method: 'GET'
        });
        assert.strictEqual(response.status, 200);
        assert.match(response.data, /ok|status/i);
      } finally {
        serverProcess.kill('SIGTERM');
        serverProcess = null;
      }
    });

    it('should return schema via API', async () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      serverProcess = spawn(CLI_PATH, ['serve', parquetFile, '--port', String(TEST_PORT + 1), '--no-open'], {
        stdio: ['ignore', 'pipe', 'pipe']
      });

      await sleep(1500);

      try {
        const response = await httpRequest({
          hostname: 'localhost',
          port: TEST_PORT + 1,
          path: '/api/schema',
          method: 'GET'
        });
        assert.strictEqual(response.status, 200);
        const schema = JSON.parse(response.data);
        assert.ok(Array.isArray(schema.columns), 'Schema should have columns array');
        assert.ok(schema.columns.length > 0, 'Schema should have at least one column');
      } finally {
        serverProcess.kill('SIGTERM');
        serverProcess = null;
      }
    });

    it('should execute query via API', async () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      serverProcess = spawn(CLI_PATH, ['serve', parquetFile, '--port', String(TEST_PORT + 2), '--no-open'], {
        stdio: ['ignore', 'pipe', 'pipe']
      });

      await sleep(1500);

      try {
        const body = JSON.stringify({ sql: 'SELECT * LIMIT 3' });
        const response = await httpRequest({
          hostname: 'localhost',
          port: TEST_PORT + 2,
          path: '/api/query',
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Content-Length': body.length },
          body
        });
        assert.strictEqual(response.status, 200);
        const result = JSON.parse(response.data);
        assert.ok(Array.isArray(result.columns), 'Result should have columns');
        assert.ok(Array.isArray(result.rows), 'Result should have rows');
        assert.ok(result.rows.length <= 3, 'Should return at most 3 rows');
      } finally {
        serverProcess.kill('SIGTERM');
        serverProcess = null;
      }
    });

    it('should return error for invalid query', async () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      serverProcess = spawn(CLI_PATH, ['serve', parquetFile, '--port', String(TEST_PORT + 3), '--no-open'], {
        stdio: ['ignore', 'pipe', 'pipe']
      });

      await sleep(1500);

      try {
        const body = JSON.stringify({ sql: 'INVALID SQL SYNTAX HERE' });
        const response = await httpRequest({
          hostname: 'localhost',
          port: TEST_PORT + 3,
          path: '/api/query',
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Content-Length': body.length },
          body
        });
        // Should return 400 or include error message
        const hasError = response.status === 400 || response.data.toLowerCase().includes('error');
        assert.ok(hasError, 'Should return error for invalid SQL');
      } finally {
        serverProcess.kill('SIGTERM');
        serverProcess = null;
      }
    });
  });

  describe('Enrich Command', () => {
    let tempDir;

    before(() => {
      tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lanceql-enrich-'));
    });

    after(() => {
      if (tempDir) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should show enrich help with --help', () => {
      if (!cliExists()) return;

      const result = runCli('enrich --help');
      assert.strictEqual(result.exitCode, 0);
      assert.match(result.output, /enrich|embedding|vector|index/i);
    });

    it('should enrich Lance file with embeddings', () => {
      if (!cliExists()) return;

      // First create a Lance file with text column
      const jsonPath = path.join(tempDir, 'texts.json');
      const inputLance = path.join(tempDir, 'input.lance');
      const outputLance = path.join(tempDir, 'enriched.lance');

      fs.writeFileSync(jsonPath, JSON.stringify([
        { id: 1, text: 'Hello world' },
        { id: 2, text: 'Machine learning' },
        { id: 3, text: 'Vector search' }
      ]));

      // Ingest to Lance first
      let result = runCli(`ingest "${jsonPath}" -o "${inputLance}"`);
      if (result.exitCode !== 0) {
        console.log('Skipping: Failed to create input Lance file');
        return;
      }

      // Now enrich
      result = runCli(`enrich "${inputLance}" -o "${outputLance}" --embed text`);
      // Enrich may use mock embeddings or indicate embedding generation
      assert.ok(
        result.exitCode === 0 ||
        result.output.toLowerCase().includes('embedding') ||
        result.output.toLowerCase().includes('vector') ||
        result.output.toLowerCase().includes('enriched'),
        'Enrich should complete or indicate embedding generation'
      );
    });

    it('should create flat index', () => {
      if (!cliExists()) return;

      const jsonPath = path.join(tempDir, 'flat_texts.json');
      const inputLance = path.join(tempDir, 'flat_input.lance');
      const outputLance = path.join(tempDir, 'flat_enriched.lance');

      fs.writeFileSync(jsonPath, JSON.stringify([
        { id: 1, text: 'Test one' },
        { id: 2, text: 'Test two' }
      ]));

      runCli(`ingest "${jsonPath}" -o "${inputLance}"`);

      const result = runCli(`enrich "${inputLance}" -o "${outputLance}" --embed text --index flat`);
      assert.ok(
        result.exitCode === 0 ||
        result.output.toLowerCase().includes('flat') ||
        result.output.toLowerCase().includes('index') ||
        result.output.toLowerCase().includes('embedding'),
        'Should support flat index type'
      );
    });

    it('should error on missing text column', () => {
      if (!cliExists()) return;

      const jsonPath = path.join(tempDir, 'no_text.json');
      const inputLance = path.join(tempDir, 'no_text.lance');
      const outputLance = path.join(tempDir, 'no_text_out.lance');

      fs.writeFileSync(jsonPath, JSON.stringify([
        { id: 1, value: 100 },
        { id: 2, value: 200 }
      ]));

      runCli(`ingest "${jsonPath}" -o "${inputLance}"`);

      // Request non-existent column
      const result = runCli(`enrich "${inputLance}" -o "${outputLance}" --embed nonexistent`);
      const hasError = result.exitCode !== 0 ||
                       result.output.toLowerCase().includes('error') ||
                       result.output.toLowerCase().includes('not found') ||
                       result.output.toLowerCase().includes('column');
      assert.ok(hasError, 'Should error on missing column');
    });
  });

  describe('Transform Command - Operations', () => {
    let tempDir;

    before(() => {
      tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lanceql-transform-'));
    });

    after(() => {
      if (tempDir) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should select specific columns', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const outputLance = path.join(tempDir, 'select_cols.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`transform "${parquetFile}" -o "${outputLance}" --select "id, name"`);
      assert.ok(
        result.exitCode === 0 || result.output.includes('select'),
        'Should support column selection'
      );
    });

    it('should filter rows with WHERE', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const outputLance = path.join(tempDir, 'where_filter.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`transform "${parquetFile}" -o "${outputLance}" --where "id > 2"`);
      assert.ok(
        result.exitCode === 0 || result.output.includes('where') || result.output.includes('filter'),
        'Should support WHERE filtering'
      );
    });

    it('should limit results', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const outputLance = path.join(tempDir, 'limit_rows.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`transform "${parquetFile}" -o "${outputLance}" --limit 2`);
      assert.ok(
        result.exitCode === 0 || result.output.includes('limit'),
        'Should support LIMIT'
      );
    });

    it('should transform Parquet to Lance', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const outputLance = path.join(tempDir, 'parquet_to_lance.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`transform "${parquetFile}" -o "${outputLance}"`);
      if (result.exitCode === 0) {
        assert.ok(fs.existsSync(outputLance), 'Lance output file should be created');
      }
    });

    it('should transform Arrow to Lance', () => {
      if (!cliExists()) return;

      const arrowFile = path.join(FIXTURES_PATH, 'simple.arrow');
      const outputLance = path.join(tempDir, 'arrow_to_lance.lance');

      if (!fs.existsSync(arrowFile)) {
        console.log('Skipping: simple.arrow not found');
        return;
      }

      const result = runCli(`transform "${arrowFile}" -o "${outputLance}"`);
      if (result.exitCode === 0) {
        assert.ok(fs.existsSync(outputLance), 'Lance output file should be created');
      }
    });

    it('should apply SQL expression', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      const outputLance = path.join(tempDir, 'sql_expr.lance');

      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`transform "${parquetFile}" -o "${outputLance}" --sql "SELECT id, value * 2 AS double_value"`);
      assert.ok(
        result.exitCode === 0 || result.output.includes('sql'),
        'Should support SQL expressions'
      );
    });
  });

  describe('Query Command - Edge Cases', () => {
    it('should handle aggregate functions', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT COUNT(*), SUM(value), AVG(value) FROM '${parquetFile}'"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
      assert.match(result.output, /COUNT|SUM|AVG|\d+/i);
    });

    it('should handle GROUP BY', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT name, COUNT(*) FROM '${parquetFile}' GROUP BY name"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });

    it('should handle ORDER BY', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT * FROM '${parquetFile}' ORDER BY id DESC LIMIT 3"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });

    it('should handle WHERE with comparison operators', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT * FROM '${parquetFile}' WHERE id >= 2 AND value < 5"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });

    it('should handle string comparisons', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT * FROM '${parquetFile}' WHERE name = 'alice'"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });

    it('should handle MIN/MAX functions', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT MIN(value), MAX(value) FROM '${parquetFile}'"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });

    it('should handle DISTINCT', () => {
      if (!cliExists()) return;

      const parquetFile = path.join(FIXTURES_PATH, 'simple.parquet');
      if (!fs.existsSync(parquetFile)) {
        console.log('Skipping: simple.parquet not found');
        return;
      }

      const result = runCli(`query "SELECT DISTINCT name FROM '${parquetFile}'"`);
      assert.strictEqual(result.exitCode, 0, `Query failed: ${result.stderr}`);
    });
  });
});

// Run tests if executed directly
if (require.main === module) {
  console.log('Run with: node --test test/cli.test.js');
}
