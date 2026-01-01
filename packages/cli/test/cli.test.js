/**
 * LanceQL CLI Tests
 *
 * Tests the CLI when run via npx/bunx:
 *   npx @metal0/lanceql --version
 *   npx @metal0/lanceql query "SELECT * FROM 'file.parquet'"
 *   npx @metal0/lanceql ingest data.csv -o out.lance
 */

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');
const { describe, it, before, after } = require('node:test');

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
});

// Run tests if executed directly
if (require.main === module) {
  console.log('Run with: node --test test/cli.test.js');
}
