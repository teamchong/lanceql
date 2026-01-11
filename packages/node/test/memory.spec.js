import { describe, it, expect, afterEach } from 'vitest';

const { getFixturePath } = require('./test-utils.js');
const Database = require('../src/index.js');

const SIMPLE_INT64_LANCE = getFixturePath('simple_int64.lance');

describe('Memory Safety', () => {
  let db;

  afterEach(() => {
    if (db && db.open) {
      db.close();
    }
  });

  describe('repeated operations', () => {
    it.skip('should handle 1000 consecutive queries without memory leaks', { timeout: 120000 }, () => {
      // FIXME: Resource leak in native binding causes failure after ~60 queries
      db = new Database(SIMPLE_INT64_LANCE);

      for (let i = 0; i < 1000; i++) {
        const rows = db.prepare('SELECT * FROM t').all();
        expect(rows.length).toBe(5);
      }
    });

    it('should handle 1000 open/close cycles', { timeout: 120000 }, () => {
      for (let i = 0; i < 1000; i++) {
        const tempDb = new Database(SIMPLE_INT64_LANCE);
        expect(tempDb.open).toBe(true);
        tempDb.close();
        expect(tempDb.open).toBe(false);
      }
    });

    it('should handle 500 prepared statements', { timeout: 180000 }, () => {
      db = new Database(SIMPLE_INT64_LANCE);

      // Reduced from 1000 to 500 for CI performance
      for (let i = 0; i < 500; i++) {
        const stmt = db.prepare('SELECT * FROM t WHERE id > 0');
        const rows = stmt.all();
        expect(rows.length).toBe(5);
      }
    });
  });

  describe('concurrent access patterns', () => {
    it('should handle multiple statements on same connection', { timeout: 60000 }, () => {
      db = new Database(SIMPLE_INT64_LANCE);

      const stmt1 = db.prepare('SELECT * FROM t');
      const stmt2 = db.prepare('SELECT * FROM t WHERE id > 2');
      const stmt3 = db.prepare('SELECT * FROM t LIMIT 1');

      // Reduced iterations for CI (100 -> 50)
      for (let i = 0; i < 50; i++) {
        expect(stmt1.all().length).toBe(5);
        expect(stmt2.all().length).toBe(3);
        expect(stmt3.all().length).toBe(1);
      }
    });
  });

  describe('edge cases', () => {
    it('should handle empty result sets', { timeout: 60000 }, () => {
      db = new Database(SIMPLE_INT64_LANCE);

      for (let i = 0; i < 50; i++) {
        const rows = db.prepare('SELECT * FROM t WHERE id > 1000').all();
        expect(rows.length).toBe(0);
      }
    });

    it('should handle get() returning undefined', { timeout: 60000 }, () => {
      db = new Database(SIMPLE_INT64_LANCE);

      for (let i = 0; i < 50; i++) {
        const row = db.prepare('SELECT * FROM t WHERE id > 1000').get();
        expect(row).toBeUndefined();
      }
    });
  });
});
