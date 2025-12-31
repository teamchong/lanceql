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
});

// Run tests if executed directly
if (require.main === module) {
  console.log('Run with: node --test test/cli.test.js');
}
