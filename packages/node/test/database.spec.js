import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';

const { getFixturePath } = require('./test-utils.js');
const Database = require('../src/index.js');

// Fixtures
const SIMPLE_INT64_LANCE = getFixturePath('simple_int64.lance');

describe('Database', () => {
  let db;

  afterEach(() => {
    if (db && db.open) {
      db.close();
    }
  });

  describe('constructor', () => {
    it('should open a Lance file by path', () => {
      db = new Database(SIMPLE_INT64_LANCE);
      expect(db).toBeDefined();
      expect(db.open).toBe(true);
    });

    it('should throw for invalid paths', () => {
      expect(() => new Database('/nonexistent/path.lance')).toThrow();
    });

    it('should throw for :memory: (not supported)', () => {
      expect(() => new Database(':memory:')).toThrow('In-memory databases not supported');
    });
  });

  describe('close()', () => {
    it('should close the database', () => {
      db = new Database(SIMPLE_INT64_LANCE);
      expect(db.open).toBe(true);
      db.close();
      expect(db.open).toBe(false);
    });

    it('should be safe to call multiple times', () => {
      db = new Database(SIMPLE_INT64_LANCE);
      db.close();
      expect(() => db.close()).not.toThrow();
    });
  });

  describe('name property', () => {
    it('should return the database path', () => {
      db = new Database(SIMPLE_INT64_LANCE);
      expect(db.name).toBe(SIMPLE_INT64_LANCE);
    });
  });

  describe('readonly property', () => {
    it('should be true for Lance files', () => {
      db = new Database(SIMPLE_INT64_LANCE);
      expect(db.readonly).toBe(true);
    });
  });
});

describe('Statement', () => {
  let db;

  beforeAll(() => {
    db = new Database(SIMPLE_INT64_LANCE);
  });

  afterAll(() => {
    if (db && db.open) {
      db.close();
    }
  });

  describe('prepare()', () => {
    it('should create a statement from SQL', () => {
      const stmt = db.prepare('SELECT * FROM t');
      expect(stmt).toBeDefined();
    });

    it('should throw for invalid SQL', () => {
      expect(() => db.prepare('INVALID SQL')).toThrow();
    });
  });

  describe('all()', () => {
    it('should return all rows as an array', () => {
      const rows = db.prepare('SELECT * FROM t').all();
      expect(Array.isArray(rows)).toBe(true);
      expect(rows.length).toBe(5);
    });

    it('should return objects with column names', () => {
      const rows = db.prepare('SELECT * FROM t').all();
      expect(rows[0]).toHaveProperty('id');
      expect(rows[0].id).toBe(1);
    });
  });

  describe('get()', () => {
    it('should return the first row', () => {
      const row = db.prepare('SELECT * FROM t').get();
      expect(row).toBeDefined();
      expect(row.id).toBe(1);
    });

    it('should return undefined for empty results', () => {
      const row = db.prepare('SELECT * FROM t WHERE id > 1000').get();
      expect(row).toBeUndefined();
    });
  });

  describe('run()', () => {
    it('should return {changes: 0, lastInsertRowid: 0} for read-only db', () => {
      const result = db.prepare('SELECT * FROM t').run();
      expect(result.changes).toBe(0);
      expect(result.lastInsertRowid).toBe(0);
    });
  });
});

describe('SQL Queries', () => {
  let db;

  beforeAll(() => {
    db = new Database(SIMPLE_INT64_LANCE);
  });

  afterAll(() => {
    if (db && db.open) {
      db.close();
    }
  });

  describe('WHERE clause', () => {
    it('should filter with > operator', () => {
      const rows = db.prepare('SELECT * FROM t WHERE id > 2').all();
      expect(rows.length).toBe(3);
      expect(rows[0].id).toBe(3);
    });

    it('should filter with = operator', () => {
      const rows = db.prepare('SELECT * FROM t WHERE id = 3').all();
      expect(rows.length).toBe(1);
      expect(rows[0].id).toBe(3);
    });

    it('should filter with < operator', () => {
      const rows = db.prepare('SELECT * FROM t WHERE id < 3').all();
      expect(rows.length).toBe(2);
    });
  });

  describe('LIMIT clause', () => {
    it('should limit results', () => {
      const rows = db.prepare('SELECT * FROM t LIMIT 2').all();
      expect(rows.length).toBe(2);
    });
  });
});

describe('Error Handling', () => {
  let db;

  beforeAll(() => {
    db = new Database(SIMPLE_INT64_LANCE);
  });

  afterAll(() => {
    if (db && db.open) {
      db.close();
    }
  });

  describe('SqliteError', () => {
    it('should have code property', () => {
      try {
        db.prepare('INVALID');
      } catch (err) {
        expect(err.code).toBeDefined();
      }
    });
  });
});
