const Database = require('../src/index.js');
const assert = require('assert');
const { getFixturePath, getBetterSqlite3FixturePath } = require('./test-utils.js');

// Test with existing fixture
const dbPath = getFixturePath('simple_int64.lance');
const stringDbPath = getBetterSqlite3FixturePath('simple.lance');

console.log('=== Parameter Binding Tests ===\n');

// Test 1: Single integer parameter
console.log('Test 1: Single integer parameter (WHERE id = ?)');
try {
    const db = new Database(dbPath);
    const stmt = db.prepare('SELECT * FROM t WHERE id = ?');
    const rows = stmt.all(3);
    console.log(`Rows: ${JSON.stringify(rows)}`);
    assert.strictEqual(rows.length, 1);
    assert.strictEqual(rows[0].id, 3);
    console.log('✓ Single integer parameter works');
    db.close();
} catch (err) {
    console.error('✗ Single integer parameter failed:', err.message);
    process.exit(1);
}

// Test 2: Multiple integer parameters
console.log('\nTest 2: Multiple integer parameters (WHERE id > ? AND id < ?)');
try {
    const db = new Database(dbPath);
    const stmt = db.prepare('SELECT * FROM t WHERE id > ? AND id < ?');
    const rows = stmt.all(2, 5);
    console.log(`Rows: ${JSON.stringify(rows)}`);
    assert.strictEqual(rows.length, 2);
    assert.strictEqual(rows[0].id, 3);
    assert.strictEqual(rows[1].id, 4);
    console.log('✓ Multiple integer parameters work');
    db.close();
} catch (err) {
    console.error('✗ Multiple integer parameters failed:', err.message);
    process.exit(1);
}

// Test 3: String parameter
console.log('\nTest 3: String parameter (WHERE a = ?)');
try {
    const db = new Database(stringDbPath);
    const stmt = db.prepare('SELECT b FROM t WHERE a = ?');
    const rows = stmt.all('foo');
    console.log(`Rows: ${JSON.stringify(rows)}`);
    assert.strictEqual(rows.length, 1);
    assert.strictEqual(rows[0].b, 1);
    console.log('✓ String parameter works');
    db.close();
} catch (err) {
    console.error('✗ String parameter failed:', err.message);
    process.exit(1);
}

// Test 4: Bind method
console.log('\nTest 4: Using bind() method');
try {
    const db = new Database(dbPath);
    const stmt = db.prepare('SELECT * FROM t WHERE id = ?').bind(4);
    const rows = stmt.all();
    console.log(`Rows: ${JSON.stringify(rows)}`);
    assert.strictEqual(rows.length, 1);
    assert.strictEqual(rows[0].id, 4);
    console.log('✓ bind() method works');
    db.close();
} catch (err) {
    console.error('✗ bind() method failed:', err.message);
    process.exit(1);
}

// Test 5: get() with parameter
console.log('\nTest 5: get() with parameter');
try {
    const db = new Database(dbPath);
    const stmt = db.prepare('SELECT * FROM t WHERE id = ?');
    const row = stmt.get(2);
    console.log(`Row: ${JSON.stringify(row)}`);
    assert.strictEqual(row.id, 2);
    console.log('✓ get() with parameter works');
    db.close();
} catch (err) {
    console.error('✗ get() with parameter failed:', err.message);
    process.exit(1);
}

// Test 6: No parameters still works
console.log('\nTest 6: Query without parameters still works');
try {
    const db = new Database(dbPath);
    const stmt = db.prepare('SELECT * FROM t');
    const rows = stmt.all();
    console.log(`Rows: ${JSON.stringify(rows)}`);
    assert.strictEqual(rows.length, 5);
    console.log('✓ Query without parameters still works');
    db.close();
} catch (err) {
    console.error('✗ Query without parameters failed:', err.message);
    process.exit(1);
}

console.log('\n✅ All parameter binding tests passed!');
