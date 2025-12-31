// @ts-check
import { test, expect } from '@playwright/test';

test.describe('Vault SQL Operations', () => {
    test.beforeEach(async ({ page }) => {
        // Navigate to test page
        await page.goto('/test-vault-sql.html');

        // Wait for page to load
        await page.waitForLoadState('domcontentloaded');
    });

    test('vault initializes successfully', async ({ page }) => {
        // Initialize vault via page context
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            try {
                const v = await vault();
                return { success: true };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        expect(result.success).toBe(true);
    });

    test('CREATE TABLE operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Test 1: Basic CREATE TABLE
            try {
                await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
                tests.push({ sql: 'CREATE TABLE test_users', pass: true });
            } catch (e) {
                tests.push({ sql: 'CREATE TABLE test_users', pass: false, error: e.message });
            }

            // Test 2: CREATE TABLE IF NOT EXISTS
            try {
                await v.exec('CREATE TABLE IF NOT EXISTS test_users (id INT)');
                tests.push({ sql: 'CREATE TABLE IF NOT EXISTS', pass: true });
            } catch (e) {
                tests.push({ sql: 'CREATE TABLE IF NOT EXISTS', pass: false, error: e.message });
            }

            // Test 3: Create another table
            try {
                await v.exec('CREATE TABLE test_products (id INT, name TEXT, price FLOAT)');
                tests.push({ sql: 'CREATE TABLE test_products', pass: true });
            } catch (e) {
                tests.push({ sql: 'CREATE TABLE test_products', pass: false, error: e.message });
            }

            // Cleanup
            try {
                await v.exec('DROP TABLE IF EXISTS test_users');
                await v.exec('DROP TABLE IF EXISTS test_products');
            } catch (e) { /* ignore cleanup errors */ }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('INSERT operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');

            // Test 1: Single row insert
            try {
                await v.exec("INSERT INTO test_users (id, name, age) VALUES (1, 'Alice', 30)");
                tests.push({ sql: 'INSERT single row', pass: true });
            } catch (e) {
                tests.push({ sql: 'INSERT single row', pass: false, error: e.message });
            }

            // Test 2: Multiple row insert
            try {
                await v.exec("INSERT INTO test_users VALUES (2, 'Bob', 25), (3, 'Charlie', 35)");
                tests.push({ sql: 'INSERT multiple rows', pass: true });
            } catch (e) {
                tests.push({ sql: 'INSERT multiple rows', pass: false, error: e.message });
            }

            // Verify count
            try {
                const res = await v.exec('SELECT COUNT(*) FROM test_users');
                const count = Object.values(res.rows?.[0] || {})[0];
                tests.push({ sql: 'Verify count=3', pass: count === 3, error: `got ${count}` });
            } catch (e) {
                tests.push({ sql: 'Verify count', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('SELECT basic operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)");

            // Test 1: SELECT *
            try {
                const res = await v.exec('SELECT * FROM test_users');
                tests.push({ sql: 'SELECT *', pass: res.rows?.length === 3, error: `got ${res.rows?.length} rows` });
            } catch (e) {
                tests.push({ sql: 'SELECT *', pass: false, error: e.message });
            }

            // Test 2: SELECT specific columns
            try {
                const res = await v.exec('SELECT name, age FROM test_users');
                tests.push({ sql: 'SELECT name, age', pass: res.columns?.length === 2, error: `got ${res.columns?.length} cols` });
            } catch (e) {
                tests.push({ sql: 'SELECT name, age', pass: false, error: e.message });
            }

            // Test 3: SELECT with WHERE
            try {
                const res = await v.exec('SELECT * FROM test_users WHERE id = 1');
                tests.push({ sql: 'SELECT WHERE id=1', pass: res.rows?.length === 1, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'SELECT WHERE id=1', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('WHERE operators', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35), (4, 'Diana', 28)");

            // Test operators
            const testCases = [
                { sql: 'SELECT * FROM test_users WHERE age > 28', expectRows: 2 },
                { sql: 'SELECT * FROM test_users WHERE age <= 28', expectRows: 2 },
                { sql: "SELECT * FROM test_users WHERE name != 'Alice'", expectRows: 3 },
                { sql: "SELECT * FROM test_users WHERE name LIKE 'A%'", expectRows: 1 },
                { sql: 'SELECT * FROM test_users WHERE age BETWEEN 25 AND 30', expectRows: 3 },
                { sql: 'SELECT * FROM test_users WHERE id IN (1, 2)', expectRows: 2 },
            ];

            for (const tc of testCases) {
                try {
                    const res = await v.exec(tc.sql);
                    const pass = res.rows?.length === tc.expectRows;
                    tests.push({ sql: tc.sql, pass, error: `expected ${tc.expectRows}, got ${res.rows?.length}` });
                } catch (e) {
                    tests.push({ sql: tc.sql, pass: false, error: e.message });
                }
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('AND/OR logic operators', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35), (4, 'Diana', 28)");

            // Test AND
            try {
                const res = await v.exec('SELECT * FROM test_users WHERE age > 25 AND age < 35');
                tests.push({ sql: 'WHERE age > 25 AND age < 35', pass: res.rows?.length === 2, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'AND operator', pass: false, error: e.message });
            }

            // Test OR
            try {
                const res = await v.exec('SELECT * FROM test_users WHERE age < 26 OR age > 34');
                tests.push({ sql: 'WHERE age < 26 OR age > 34', pass: res.rows?.length === 2, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'OR operator', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('UPDATE operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25)");

            // Test UPDATE
            try {
                await v.exec("UPDATE test_users SET age = 31 WHERE name = 'Alice'");
                const res = await v.exec("SELECT age FROM test_users WHERE name = 'Alice'");
                const age = res.rows?.[0]?.age;
                tests.push({ sql: 'UPDATE age', pass: age === 31, error: `expected 31, got ${age}` });
            } catch (e) {
                tests.push({ sql: 'UPDATE', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('DELETE operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)");

            // Test DELETE
            try {
                await v.exec('DELETE FROM test_users WHERE id = 3');
                const res = await v.exec('SELECT COUNT(*) FROM test_users');
                const count = Object.values(res.rows?.[0] || {})[0];
                tests.push({ sql: 'DELETE WHERE id=3', pass: count === 2, error: `expected 2, got ${count}` });
            } catch (e) {
                tests.push({ sql: 'DELETE', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Aggregation functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 20), (3, 'Charlie', 40)");

            // Helper to get first value from result row
            const getFirstValue = (res) => Object.values(res.rows?.[0] || {})[0];

            // Test COUNT
            try {
                const res = await v.exec('SELECT COUNT(*) FROM test_users');
                const val = getFirstValue(res);
                tests.push({ sql: 'COUNT(*)', pass: val === 3, error: `got ${val}` });
            } catch (e) {
                tests.push({ sql: 'COUNT(*)', pass: false, error: e.message });
            }

            // Test SUM
            try {
                const res = await v.exec('SELECT SUM(age) FROM test_users');
                const val = getFirstValue(res);
                tests.push({ sql: 'SUM(age)', pass: val === 90, error: `got ${val}` });
            } catch (e) {
                tests.push({ sql: 'SUM(age)', pass: false, error: e.message });
            }

            // Test AVG
            try {
                const res = await v.exec('SELECT AVG(age) FROM test_users');
                const val = getFirstValue(res);
                tests.push({ sql: 'AVG(age)', pass: val === 30, error: `got ${val}` });
            } catch (e) {
                tests.push({ sql: 'AVG(age)', pass: false, error: e.message });
            }

            // Test MIN
            try {
                const res = await v.exec('SELECT MIN(age) FROM test_users');
                const val = getFirstValue(res);
                tests.push({ sql: 'MIN(age)', pass: val === 20, error: `got ${val}` });
            } catch (e) {
                tests.push({ sql: 'MIN(age)', pass: false, error: e.message });
            }

            // Test MAX
            try {
                const res = await v.exec('SELECT MAX(age) FROM test_users');
                const val = getFirstValue(res);
                tests.push({ sql: 'MAX(age)', pass: val === 40, error: `got ${val}` });
            } catch (e) {
                tests.push({ sql: 'MAX(age)', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('GROUP BY and HAVING', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 30), (3, 'Charlie', 25)");

            // Test GROUP BY
            try {
                const res = await v.exec('SELECT age, COUNT(*) FROM test_users GROUP BY age');
                tests.push({ sql: 'GROUP BY age', pass: res.rows?.length === 2, error: `got ${res.rows?.length} groups` });
            } catch (e) {
                tests.push({ sql: 'GROUP BY', pass: false, error: e.message });
            }

            // Test HAVING
            try {
                const res = await v.exec('SELECT age, COUNT(*) FROM test_users GROUP BY age HAVING COUNT(*) > 1');
                tests.push({ sql: 'HAVING COUNT(*) > 1', pass: res.rows?.length === 1, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'HAVING', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('ORDER BY, LIMIT, OFFSET', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35), (4, 'Diana', 28)");

            // Test ORDER BY ASC
            try {
                const res = await v.exec('SELECT name FROM test_users ORDER BY age ASC');
                const first = res.rows?.[0]?.name;
                tests.push({ sql: 'ORDER BY age ASC', pass: first === 'Bob', error: `first: ${first}` });
            } catch (e) {
                tests.push({ sql: 'ORDER BY ASC', pass: false, error: e.message });
            }

            // Test ORDER BY DESC
            try {
                const res = await v.exec('SELECT name FROM test_users ORDER BY age DESC');
                const first = res.rows?.[0]?.name;
                tests.push({ sql: 'ORDER BY age DESC', pass: first === 'Charlie', error: `first: ${first}` });
            } catch (e) {
                tests.push({ sql: 'ORDER BY DESC', pass: false, error: e.message });
            }

            // Test LIMIT
            try {
                const res = await v.exec('SELECT * FROM test_users LIMIT 2');
                tests.push({ sql: 'LIMIT 2', pass: res.rows?.length === 2, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'LIMIT', pass: false, error: e.message });
            }

            // Test LIMIT with OFFSET
            try {
                const res = await v.exec('SELECT * FROM test_users LIMIT 2 OFFSET 1');
                tests.push({ sql: 'LIMIT 2 OFFSET 1', pass: res.rows?.length === 2, error: `got ${res.rows?.length}` });
            } catch (e) {
                tests.push({ sql: 'LIMIT OFFSET', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('JOIN operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_users (id INT, name TEXT)');
            await v.exec('CREATE TABLE test_orders (id INT, user_id INT, product TEXT)');
            await v.exec("INSERT INTO test_users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')");
            await v.exec("INSERT INTO test_orders VALUES (1, 1, 'Laptop'), (2, 1, 'Phone'), (3, 2, 'Tablet')");

            // Test INNER JOIN
            try {
                const res = await v.exec('SELECT u.name, o.product FROM test_users u JOIN test_orders o ON u.id = o.user_id');
                tests.push({ sql: 'INNER JOIN', pass: res.rows?.length === 3, error: `got ${res.rows?.length} rows` });
            } catch (e) {
                tests.push({ sql: 'INNER JOIN', pass: false, error: e.message });
            }

            // Test LEFT JOIN
            try {
                const res = await v.exec('SELECT u.name, o.product FROM test_users u LEFT JOIN test_orders o ON u.id = o.user_id');
                // Charlie has no orders, should appear with null product
                tests.push({ sql: 'LEFT JOIN', pass: res.rows?.length >= 3, error: `got ${res.rows?.length} rows` });
            } catch (e) {
                tests.push({ sql: 'LEFT JOIN', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_users');
            await v.exec('DROP TABLE IF EXISTS test_orders');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('DROP TABLE operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE test_drop (id INT)');

            // Test DROP TABLE
            try {
                await v.exec('DROP TABLE test_drop');
                tests.push({ sql: 'DROP TABLE', pass: true });
            } catch (e) {
                tests.push({ sql: 'DROP TABLE', pass: false, error: e.message });
            }

            // Test DROP TABLE IF EXISTS (should not error)
            try {
                await v.exec('DROP TABLE IF EXISTS nonexistent_table');
                tests.push({ sql: 'DROP TABLE IF EXISTS', pass: true });
            } catch (e) {
                tests.push({ sql: 'DROP TABLE IF EXISTS', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('NEAR clause parsing', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup table with text column
            await v.exec('CREATE TABLE test_docs (id INT, content TEXT)');
            await v.exec("INSERT INTO test_docs VALUES (1, 'Hello world'), (2, 'Goodbye world')");

            // Test NEAR parsing - should parse correctly but fail due to no model loaded
            try {
                await v.exec("SELECT * FROM test_docs WHERE content NEAR 'hello'");
                tests.push({ sql: 'NEAR basic', pass: false, error: 'Expected error - no model loaded' });
            } catch (e) {
                // Expected to fail with model error, not parse error
                const isModelError = e.message.includes('model') || e.message.includes('NEAR');
                tests.push({ sql: 'NEAR basic', pass: isModelError, error: e.message });
            }

            // Test NEAR with TOPK parsing
            try {
                await v.exec("SELECT * FROM test_docs WHERE content NEAR 'hello' TOPK 5");
                tests.push({ sql: 'NEAR TOPK', pass: false, error: 'Expected error - no model loaded' });
            } catch (e) {
                const isModelError = e.message.includes('model') || e.message.includes('NEAR');
                tests.push({ sql: 'NEAR TOPK', pass: isModelError, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE IF EXISTS test_docs');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('DISTINCT operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup: Create table with duplicate values
            await v.exec('DROP TABLE IF EXISTS users');
            await v.exec('CREATE TABLE users (id INTEGER, name TEXT, age INTEGER, city TEXT)');
            await v.exec("INSERT INTO users VALUES (1, 'Alice', 30, 'NYC')");
            await v.exec("INSERT INTO users VALUES (2, 'Bob', 25, 'LA')");
            await v.exec("INSERT INTO users VALUES (3, 'Charlie', 30, 'NYC')");
            await v.exec("INSERT INTO users VALUES (4, 'Diana', 25, 'Chicago')");
            await v.exec("INSERT INTO users VALUES (5, 'Eve', 30, 'NYC')");

            // Test DISTINCT on single column
            const distinctAge = await v.exec('SELECT DISTINCT age FROM users');
            const ages = distinctAge.rows.map(r => r.age).sort((a, b) => a - b);
            tests.push({
                sql: 'DISTINCT single column',
                pass: ages.length === 2 && ages[0] === 25 && ages[1] === 30,
                error: `Expected [25, 30], got ${JSON.stringify(ages)}`
            });

            // Test DISTINCT on single column (city)
            const distinctCity = await v.exec('SELECT DISTINCT city FROM users');
            const cities = distinctCity.rows.map(r => r.city).sort();
            tests.push({
                sql: 'DISTINCT cities',
                pass: cities.length === 3 && cities.includes('NYC') && cities.includes('LA') && cities.includes('Chicago'),
                error: `Expected ['Chicago', 'LA', 'NYC'], got ${JSON.stringify(cities)}`
            });

            // Test DISTINCT on multiple columns
            // Data: Alice(30,NYC), Bob(25,LA), Charlie(30,NYC), Diana(25,Chicago), Eve(30,NYC)
            // Unique combos: (30,NYC), (25,LA), (25,Chicago) = 3
            const distinctMulti = await v.exec('SELECT DISTINCT age, city FROM users');
            tests.push({
                sql: 'DISTINCT multiple columns',
                pass: distinctMulti.rows.length === 3,
                error: `Expected 3 unique age/city combos, got ${distinctMulti.rows.length}`
            });

            // Cleanup
            await v.exec('DROP TABLE users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Column and table aliases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup tables
            await v.exec('DROP TABLE IF EXISTS users');
            await v.exec('DROP TABLE IF EXISTS orders');
            await v.exec('CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)');
            await v.exec('CREATE TABLE orders (id INTEGER, user_id INTEGER, product TEXT, amount INTEGER)');
            await v.exec("INSERT INTO users VALUES (1, 'Alice', 30)");
            await v.exec("INSERT INTO users VALUES (2, 'Bob', 25)");
            await v.exec("INSERT INTO orders VALUES (1, 1, 'Book', 20)");
            await v.exec("INSERT INTO orders VALUES (2, 1, 'Pen', 5)");
            await v.exec("INSERT INTO orders VALUES (3, 2, 'Notebook', 15)");

            // Test column aliases
            const colAlias = await v.exec('SELECT name AS username, age AS years FROM users WHERE id = 1');
            tests.push({
                sql: 'Column aliases',
                pass: colAlias.rows[0].username === 'Alice' && colAlias.rows[0].years === 30,
                error: `Expected {username: 'Alice', years: 30}, got ${JSON.stringify(colAlias.rows[0])}`
            });

            // Test table alias with dot notation
            const tableAlias = await v.exec('SELECT u.name, u.age FROM users u WHERE u.id = 1');
            tests.push({
                sql: 'Table alias with dot notation',
                pass: tableAlias.rows[0].name === 'Alice' && tableAlias.rows[0].age === 30,
                error: `Expected Alice/30, got ${JSON.stringify(tableAlias.rows[0])}`
            });

            // Test table alias in JOIN
            const joinAlias = await v.exec('SELECT u.name, o.product FROM users u JOIN orders o ON u.id = o.user_id WHERE u.id = 1');
            tests.push({
                sql: 'Table alias in JOIN',
                pass: joinAlias.rows.length === 2 && joinAlias.rows[0].name === 'Alice',
                error: `Expected 2 rows for Alice, got ${joinAlias.rows.length}`
            });

            // Test combined column and table aliases
            const combined = await v.exec('SELECT u.name AS customer, o.product AS item FROM users u JOIN orders o ON u.id = o.user_id');
            tests.push({
                sql: 'Combined aliases',
                pass: combined.rows.length === 3 && combined.rows[0].customer !== undefined && combined.rows[0].item !== undefined,
                error: `Expected customer/item fields, got ${JSON.stringify(combined.rows[0])}`
            });

            // Cleanup
            await v.exec('DROP TABLE users');
            await v.exec('DROP TABLE orders');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Advanced JOINs - RIGHT JOIN and multiple JOINs', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup tables
            await v.exec('DROP TABLE IF EXISTS users');
            await v.exec('DROP TABLE IF EXISTS orders');
            await v.exec('DROP TABLE IF EXISTS categories');
            await v.exec('CREATE TABLE users (id INTEGER, name TEXT)');
            await v.exec('CREATE TABLE orders (id INTEGER, user_id INTEGER, category_id INTEGER, product TEXT)');
            await v.exec('CREATE TABLE categories (id INTEGER, name TEXT)');

            await v.exec("INSERT INTO users VALUES (1, 'Alice')");
            await v.exec("INSERT INTO users VALUES (2, 'Bob')");
            await v.exec("INSERT INTO users VALUES (3, 'Charlie')"); // No orders

            await v.exec("INSERT INTO categories VALUES (1, 'Books')");
            await v.exec("INSERT INTO categories VALUES (2, 'Electronics')");
            await v.exec("INSERT INTO categories VALUES (3, 'Clothing')"); // No orders

            await v.exec("INSERT INTO orders VALUES (1, 1, 1, 'Novel')");
            await v.exec("INSERT INTO orders VALUES (2, 1, 2, 'Phone')");
            await v.exec("INSERT INTO orders VALUES (3, 2, 1, 'Textbook')");

            // Test RIGHT JOIN - all orders even if no matching user
            const rightJoin = await v.exec('SELECT u.name, o.product FROM users u RIGHT JOIN orders o ON u.id = o.user_id');
            tests.push({
                sql: 'RIGHT JOIN',
                pass: rightJoin.rows.length === 3,
                error: `Expected 3 rows (all orders), got ${rightJoin.rows.length}`
            });

            // Test multiple JOINs (3 tables)
            const multiJoin = await v.exec('SELECT u.name, o.product, c.name AS category FROM users u JOIN orders o ON u.id = o.user_id JOIN categories c ON o.category_id = c.id');
            tests.push({
                sql: 'Multiple JOINs (3 tables)',
                pass: multiJoin.rows.length === 3,
                error: `Expected 3 rows, got ${multiJoin.rows.length}`
            });

            // Verify multiple JOIN data correctness
            const aliceBooks = multiJoin.rows.filter(r => r.name === 'Alice' && r.category === 'Books');
            tests.push({
                sql: 'Multiple JOIN data correctness',
                pass: aliceBooks.length === 1 && aliceBooks[0].product === 'Novel',
                error: `Expected Alice with Novel in Books, got ${JSON.stringify(aliceBooks)}`
            });

            // Cleanup
            await v.exec('DROP TABLE users');
            await v.exec('DROP TABLE orders');
            await v.exec('DROP TABLE categories');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Multi-column ORDER BY', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup
            await v.exec('DROP TABLE IF EXISTS employees');
            await v.exec('CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)');
            await v.exec("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 80000)");
            await v.exec("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 75000)");
            await v.exec("INSERT INTO employees VALUES (3, 'Charlie', 'Sales', 60000)");
            await v.exec("INSERT INTO employees VALUES (4, 'Diana', 'Engineering', 80000)");
            await v.exec("INSERT INTO employees VALUES (5, 'Eve', 'Sales', 65000)");

            // Test ORDER BY multiple columns with mixed directions
            const mixed = await v.exec('SELECT name, department, salary FROM employees ORDER BY department ASC, salary DESC');

            // Engineering (80000, 80000, 75000), then Sales (65000, 60000)
            const names = mixed.rows.map(r => r.name);
            // Alice and Diana have same salary, so order between them may vary, but Bob should be after them
            const engRows = mixed.rows.filter(r => r.department === 'Engineering');
            const salesRows = mixed.rows.filter(r => r.department === 'Sales');

            tests.push({
                sql: 'Multi-column ORDER BY (dept ASC, salary DESC)',
                pass: mixed.rows.slice(0, 3).every(r => r.department === 'Engineering') &&
                      mixed.rows.slice(3, 5).every(r => r.department === 'Sales'),
                error: `Expected Engineering first, then Sales. Got ${JSON.stringify(names)}`
            });

            // Verify salary ordering within department
            tests.push({
                sql: 'Salary DESC within department',
                pass: engRows[2].name === 'Bob' && engRows[2].salary === 75000,
                error: `Expected Bob with 75000 last in Engineering, got ${JSON.stringify(engRows)}`
            });

            // Test all DESC
            const allDesc = await v.exec('SELECT name, salary FROM employees ORDER BY salary DESC, name DESC');
            tests.push({
                sql: 'All DESC ordering',
                pass: allDesc.rows[0].salary === 80000 && allDesc.rows[allDesc.rows.length - 1].salary === 60000,
                error: `Expected highest salary first, got ${JSON.stringify(allDesc.rows)}`
            });

            // Cleanup
            await v.exec('DROP TABLE employees');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Data type edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup
            await v.exec('DROP TABLE IF EXISTS data_types');
            await v.exec('CREATE TABLE data_types (id INTEGER, value REAL, flag INTEGER, label TEXT)');

            // Test negative numbers
            await v.exec('INSERT INTO data_types VALUES (1, -3.14, 1, \'negative\')');
            const negResult = await v.exec('SELECT value FROM data_types WHERE id = 1');
            tests.push({
                sql: 'Negative float value',
                pass: negResult.rows[0].value < 0 && Math.abs(negResult.rows[0].value + 3.14) < 0.001,
                error: `Expected -3.14, got ${negResult.rows[0].value}`
            });

            // Test positive floats
            await v.exec('INSERT INTO data_types VALUES (2, 2.718, 0, \'euler\')');
            const floatResult = await v.exec('SELECT value FROM data_types WHERE value > 2.5');
            tests.push({
                sql: 'Float comparison',
                pass: floatResult.rows.length === 1 && floatResult.rows[0].value > 2.7,
                error: `Expected euler's number, got ${JSON.stringify(floatResult.rows)}`
            });

            // Test NULL values
            await v.exec('INSERT INTO data_types VALUES (3, NULL, NULL, NULL)');
            const nullResult = await v.exec('SELECT * FROM data_types WHERE id = 3');
            tests.push({
                sql: 'NULL values',
                pass: nullResult.rows[0].value === null && nullResult.rows[0].flag === null,
                error: `Expected NULLs, got ${JSON.stringify(nullResult.rows[0])}`
            });

            // Test boolean-like values (0 and 1)
            await v.exec('INSERT INTO data_types VALUES (4, 0, 1, \'true_flag\')');
            await v.exec('INSERT INTO data_types VALUES (5, 0, 0, \'false_flag\')');
            const boolResult = await v.exec('SELECT label FROM data_types WHERE flag = 1');
            tests.push({
                sql: 'Boolean-like flag filtering',
                pass: boolResult.rows.length === 2, // id=1 and id=4 have flag=1
                error: `Expected 2 rows with flag=1, got ${boolResult.rows.length}`
            });

            // Test zero values
            await v.exec('INSERT INTO data_types VALUES (6, 0.0, 0, \'zero\')');
            const zeroResult = await v.exec('SELECT COUNT(*) AS cnt FROM data_types WHERE value = 0');
            tests.push({
                sql: 'Zero value handling',
                pass: zeroResult.rows[0].cnt >= 2,
                error: `Expected at least 2 zeros, got ${zeroResult.rows[0].cnt}`
            });

            // Test negative number comparison
            const negCompare = await v.exec('SELECT value FROM data_types WHERE value < 0');
            tests.push({
                sql: 'Negative number comparison',
                pass: negCompare.rows.length === 1 && negCompare.rows[0].value < 0,
                error: `Expected 1 negative value, got ${negCompare.rows.length}`
            });

            // Cleanup
            await v.exec('DROP TABLE data_types');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('LIKE pattern variations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup
            await v.exec('DROP TABLE IF EXISTS names');
            await v.exec('CREATE TABLE names (id INTEGER, name TEXT, email TEXT)');
            await v.exec("INSERT INTO names VALUES (1, 'Alice', 'alice@example.com')");
            await v.exec("INSERT INTO names VALUES (2, 'Bob', 'bob@test.org')");
            await v.exec("INSERT INTO names VALUES (3, 'Charlie', 'charlie@example.com')");
            await v.exec("INSERT INTO names VALUES (4, 'Diana', 'diana@test.org')");
            await v.exec("INSERT INTO names VALUES (5, 'Alex', 'alex@example.com')");
            await v.exec("INSERT INTO names VALUES (6, 'Alicia', 'alicia@company.net')");

            // Test LIKE with % prefix (ends with)
            const endsWithE = await v.exec("SELECT name FROM names WHERE name LIKE '%e'");
            tests.push({
                sql: 'LIKE ends with',
                pass: endsWithE.rows.length === 2, // Alice, Charlie
                error: `Expected 2 names ending with 'e', got ${endsWithE.rows.length}: ${JSON.stringify(endsWithE.rows)}`
            });

            // Test LIKE with % suffix (starts with)
            const startsWithAl = await v.exec("SELECT name FROM names WHERE name LIKE 'Al%'");
            tests.push({
                sql: 'LIKE starts with',
                pass: startsWithAl.rows.length === 3, // Alice, Alex, Alicia
                error: `Expected 3 names starting with 'Al', got ${startsWithAl.rows.length}`
            });

            // Test LIKE with % on both sides (contains)
            const containsLi = await v.exec("SELECT name FROM names WHERE name LIKE '%li%'");
            tests.push({
                sql: 'LIKE contains',
                pass: containsLi.rows.length === 3, // Alice, Charlie, Alicia
                error: `Expected 3 names containing 'li', got ${containsLi.rows.length}`
            });

            // Test LIKE with underscore (single char wildcard)
            const singleChar = await v.exec("SELECT name FROM names WHERE name LIKE 'Al_ce'");
            tests.push({
                sql: 'LIKE underscore wildcard',
                pass: singleChar.rows.length === 1 && singleChar.rows[0].name === 'Alice',
                error: `Expected Alice, got ${JSON.stringify(singleChar.rows)}`
            });

            // Test LIKE on email domain
            const exampleEmails = await v.exec("SELECT email FROM names WHERE email LIKE '%@example.com'");
            tests.push({
                sql: 'LIKE email domain',
                pass: exampleEmails.rows.length === 3,
                error: `Expected 3 example.com emails, got ${exampleEmails.rows.length}`
            });

            // Cleanup
            await v.exec('DROP TABLE names');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Complex WHERE expressions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup
            await v.exec('DROP TABLE IF EXISTS products');
            await v.exec('CREATE TABLE products (id INTEGER, name TEXT, price REAL, category TEXT, stock INTEGER)');
            await v.exec("INSERT INTO products VALUES (1, 'Apple', 1.50, 'fruit', 100)");
            await v.exec("INSERT INTO products VALUES (2, 'Banana', 0.75, 'fruit', 150)");
            await v.exec("INSERT INTO products VALUES (3, 'Carrot', 0.50, 'vegetable', 80)");
            await v.exec("INSERT INTO products VALUES (4, 'Milk', 3.00, 'dairy', 50)");
            await v.exec("INSERT INTO products VALUES (5, 'Cheese', 5.00, 'dairy', 30)");
            await v.exec("INSERT INTO products VALUES (6, 'Bread', 2.50, 'bakery', 40)");
            await v.exec("INSERT INTO products VALUES (7, 'Avocado', 2.00, 'fruit', 25)");

            // Test nested AND/OR with parentheses
            const nested = await v.exec("SELECT name FROM products WHERE (category = 'fruit' AND price > 1) OR (category = 'dairy' AND stock < 40)");
            tests.push({
                sql: 'Nested AND/OR',
                pass: nested.rows.length === 3, // Apple, Avocado, Cheese
                error: `Expected 3 products, got ${nested.rows.length}: ${JSON.stringify(nested.rows.map(r => r.name))}`
            });

            // Test complex combination with IN and comparison
            const complex = await v.exec("SELECT name FROM products WHERE category IN ('fruit', 'vegetable') AND price >= 0.75 AND stock > 50");
            tests.push({
                sql: 'IN with multiple conditions',
                pass: complex.rows.length === 2, // Apple (100 stock), Banana (150 stock)
                error: `Expected 2 products, got ${complex.rows.length}`
            });

            // Test BETWEEN with AND
            const betweenAnd = await v.exec("SELECT name FROM products WHERE price BETWEEN 1 AND 3 AND stock > 30");
            tests.push({
                sql: 'BETWEEN with AND',
                pass: betweenAnd.rows.length === 3, // Apple, Milk, Bread
                error: `Expected 3 products, got ${betweenAnd.rows.length}`
            });

            // Test LIKE with OR
            const likeOr = await v.exec("SELECT name FROM products WHERE name LIKE 'A%' OR name LIKE 'B%'");
            tests.push({
                sql: 'LIKE with OR',
                pass: likeOr.rows.length === 4, // Apple, Avocado, Banana, Bread
                error: `Expected 4 products, got ${likeOr.rows.length}`
            });

            // Test empty result with impossible condition
            const empty = await v.exec("SELECT name FROM products WHERE price < 0");
            tests.push({
                sql: 'Empty result set',
                pass: empty.rows.length === 0,
                error: `Expected 0 products, got ${empty.rows.length}`
            });

            // Cleanup
            await v.exec('DROP TABLE products');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Advanced aggregations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Setup
            await v.exec('DROP TABLE IF EXISTS sales');
            await v.exec('CREATE TABLE sales (id INTEGER, region TEXT, product TEXT, amount REAL, quantity INTEGER)');
            await v.exec("INSERT INTO sales VALUES (1, 'North', 'Widget', 100.00, 10)");
            await v.exec("INSERT INTO sales VALUES (2, 'North', 'Gadget', 150.00, 5)");
            await v.exec("INSERT INTO sales VALUES (3, 'South', 'Widget', 80.00, 8)");
            await v.exec("INSERT INTO sales VALUES (4, 'South', 'Gadget', 200.00, 10)");
            await v.exec("INSERT INTO sales VALUES (5, 'East', 'Widget', 120.00, 12)");
            await v.exec("INSERT INTO sales VALUES (6, 'East', 'Gadget', 90.00, 6)");
            await v.exec("INSERT INTO sales VALUES (7, 'West', 'Widget', 110.00, 11)");

            // Test aggregate with alias
            const countAlias = await v.exec('SELECT COUNT(*) AS total_sales FROM sales');
            tests.push({
                sql: 'COUNT with alias',
                pass: countAlias.rows[0].total_sales === 7,
                error: `Expected total_sales = 7, got ${JSON.stringify(countAlias.rows[0])}`
            });

            // Test multiple aggregates in one query
            const multiAgg = await v.exec('SELECT COUNT(*) AS cnt, SUM(amount) AS total, AVG(amount) AS avg_amt FROM sales');
            tests.push({
                sql: 'Multiple aggregates',
                pass: multiAgg.rows[0].cnt === 7 && multiAgg.rows[0].total === 850,
                error: `Expected cnt=7, total=850, got ${JSON.stringify(multiAgg.rows[0])}`
            });

            // Test GROUP BY with multiple HAVING conditions
            // Data: North(250,2), South(280,2), East(210,2), West(110,1)
            // HAVING COUNT(*) >= 2 AND SUM(amount) > 200 matches North, South, East (all 3)
            const multiHaving = await v.exec('SELECT region, SUM(amount) AS total, COUNT(*) AS cnt FROM sales GROUP BY region HAVING COUNT(*) >= 2 AND SUM(amount) > 200');
            tests.push({
                sql: 'Multiple HAVING conditions',
                pass: multiHaving.rows.length === 3,
                error: `Expected 3 regions, got ${multiHaving.rows.length}: ${JSON.stringify(multiHaving.rows)}`
            });

            // Test aggregate on grouped data with ORDER BY
            const aggOrder = await v.exec('SELECT product, SUM(quantity) AS total_qty FROM sales GROUP BY product ORDER BY total_qty DESC');
            tests.push({
                sql: 'Aggregate with ORDER BY',
                pass: aggOrder.rows[0].product === 'Widget' && aggOrder.rows[0].total_qty === 41,
                error: `Expected Widget with 41 qty first, got ${JSON.stringify(aggOrder.rows)}`
            });

            // Test MIN/MAX with GROUP BY
            const minMax = await v.exec('SELECT region, MIN(amount) AS min_sale, MAX(amount) AS max_sale FROM sales GROUP BY region');
            const north = minMax.rows.find(r => r.region === 'North');
            tests.push({
                sql: 'MIN/MAX with GROUP BY',
                pass: north && north.min_sale === 100 && north.max_sale === 150,
                error: `Expected North min=100 max=150, got ${JSON.stringify(north)}`
            });

            // Test COUNT with column (non-null count)
            await v.exec("INSERT INTO sales VALUES (8, 'West', NULL, 50.00, 5)");
            const countCol = await v.exec('SELECT COUNT(product) AS product_count FROM sales');
            tests.push({
                sql: 'COUNT(column) excludes NULLs',
                pass: countCol.rows[0].product_count === 7, // 8 rows but 1 has NULL product
                error: `Expected 7 non-null products, got ${countCol.rows[0].product_count}`
            });

            // Cleanup
            await v.exec('DROP TABLE sales');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('KV basic operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Test set and get - string
            await v.set('name', 'Alice');
            const name = await v.get('name');
            tests.push({
                sql: 'set/get string',
                pass: name === 'Alice',
                error: `Expected 'Alice', got ${JSON.stringify(name)}`
            });

            // Test set and get - number
            await v.set('age', 30);
            const age = await v.get('age');
            tests.push({
                sql: 'set/get number',
                pass: age === 30,
                error: `Expected 30, got ${age}`
            });

            // Test set and get - object
            await v.set('user', { id: 1, name: 'Bob', active: true });
            const user = await v.get('user');
            tests.push({
                sql: 'set/get object',
                pass: user && user.id === 1 && user.name === 'Bob' && user.active === true,
                error: `Expected user object, got ${JSON.stringify(user)}`
            });

            // Test has - existing key
            const hasName = await v.has('name');
            tests.push({
                sql: 'has existing key',
                pass: hasName === true,
                error: `Expected true, got ${hasName}`
            });

            // Test has - non-existing key
            const hasNonExistent = await v.has('nonexistent');
            tests.push({
                sql: 'has non-existing key',
                pass: hasNonExistent === false,
                error: `Expected false, got ${hasNonExistent}`
            });

            // Test keys
            const keys = await v.keys();
            tests.push({
                sql: 'keys returns all keys',
                pass: keys.includes('name') && keys.includes('age') && keys.includes('user'),
                error: `Expected keys to include name, age, user. Got ${JSON.stringify(keys)}`
            });

            // Test delete
            const deleted = await v.delete('age');
            const ageAfterDelete = await v.get('age');
            tests.push({
                sql: 'delete key',
                pass: deleted === true && ageAfterDelete === undefined,
                error: `Expected deleted=true and value=undefined, got deleted=${deleted}, value=${ageAfterDelete}`
            });

            // Test get non-existent key returns undefined
            const missing = await v.get('doesnotexist');
            tests.push({
                sql: 'get non-existent returns undefined',
                pass: missing === undefined,
                error: `Expected undefined, got ${JSON.stringify(missing)}`
            });

            // Cleanup
            await v.delete('name');
            await v.delete('user');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('KV data types', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // String
            await v.set('str', 'hello world');
            tests.push({
                sql: 'string value',
                pass: await v.get('str') === 'hello world',
                error: `String mismatch`
            });

            // Integer
            await v.set('int', 42);
            tests.push({
                sql: 'integer value',
                pass: await v.get('int') === 42,
                error: `Integer mismatch`
            });

            // Float
            await v.set('float', 3.14159);
            const floatVal = await v.get('float');
            tests.push({
                sql: 'float value',
                pass: Math.abs(floatVal - 3.14159) < 0.0001,
                error: `Float mismatch: ${floatVal}`
            });

            // Negative number
            await v.set('neg', -100);
            tests.push({
                sql: 'negative number',
                pass: await v.get('neg') === -100,
                error: `Negative mismatch`
            });

            // Boolean true
            await v.set('bool_true', true);
            tests.push({
                sql: 'boolean true',
                pass: await v.get('bool_true') === true,
                error: `Boolean true mismatch`
            });

            // Boolean false
            await v.set('bool_false', false);
            tests.push({
                sql: 'boolean false',
                pass: await v.get('bool_false') === false,
                error: `Boolean false mismatch`
            });

            // Null
            await v.set('null_val', null);
            tests.push({
                sql: 'null value',
                pass: await v.get('null_val') === null,
                error: `Null mismatch`
            });

            // Array
            await v.set('arr', [1, 'two', { three: 3 }]);
            const arr = await v.get('arr');
            tests.push({
                sql: 'array value',
                pass: Array.isArray(arr) && arr.length === 3 && arr[0] === 1 && arr[1] === 'two' && arr[2].three === 3,
                error: `Array mismatch: ${JSON.stringify(arr)}`
            });

            // Nested object
            await v.set('nested', { a: { b: { c: 'deep' } } });
            const nested = await v.get('nested');
            tests.push({
                sql: 'nested object',
                pass: nested?.a?.b?.c === 'deep',
                error: `Nested mismatch: ${JSON.stringify(nested)}`
            });

            // Cleanup
            for (const key of await v.keys()) {
                await v.delete(key);
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('KV persistence across vault instances', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            // Create first vault and store data
            const v1 = await vault();
            await v1.set('persist_test', { value: 'should persist' });
            await v1.set('persist_num', 12345);

            // Create second vault instance (should share same storage)
            const v2 = await vault();

            // Verify data persists
            const val = await v2.get('persist_test');
            tests.push({
                sql: 'object persists',
                pass: val && val.value === 'should persist',
                error: `Expected {value: 'should persist'}, got ${JSON.stringify(val)}`
            });

            const num = await v2.get('persist_num');
            tests.push({
                sql: 'number persists',
                pass: num === 12345,
                error: `Expected 12345, got ${num}`
            });

            // Cleanup
            await v2.delete('persist_test');
            await v2.delete('persist_num');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Encryption with password', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            // Create encrypted vault with password
            const v = await vault(async () => 'test-password-123');

            // Store sensitive data
            await v.set('secret', { apiKey: 'sk-12345', token: 'jwt-abc' });
            await v.set('pin', 9999);

            // Verify data is retrievable
            const secret = await v.get('secret');
            tests.push({
                sql: 'encrypted get object',
                pass: secret && secret.apiKey === 'sk-12345' && secret.token === 'jwt-abc',
                error: `Expected secret object, got ${JSON.stringify(secret)}`
            });

            const pin = await v.get('pin');
            tests.push({
                sql: 'encrypted get number',
                pass: pin === 9999,
                error: `Expected 9999, got ${pin}`
            });

            // Create new vault with same password
            const v2 = await vault(async () => 'test-password-123');
            const secret2 = await v2.get('secret');
            tests.push({
                sql: 'same password accesses data',
                pass: secret2 && secret2.apiKey === 'sk-12345',
                error: `Expected secret with same password, got ${JSON.stringify(secret2)}`
            });

            // Cleanup
            await v2.delete('secret');
            await v2.delete('pin');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Error handling - invalid SQL', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Missing table name in SELECT
            try {
                await v.exec('SELECT FROM');
                tests.push({ sql: 'SELECT FROM (no table)', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'SELECT FROM (no table)',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            // Invalid keyword
            try {
                await v.exec('SELEKT * FROM users');
                tests.push({ sql: 'Invalid keyword SELEKT', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'Invalid keyword SELEKT',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            // Unclosed string
            try {
                await v.exec("SELECT * FROM users WHERE name = 'Alice");
                tests.push({ sql: 'Unclosed string', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'Unclosed string',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Error handling - missing table', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // SELECT from non-existent table
            try {
                await v.exec('SELECT * FROM nonexistent_table');
                tests.push({ sql: 'SELECT nonexistent table', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'SELECT nonexistent table',
                    pass: e.message.includes('not') || e.message.includes('exist') || e.message.includes('found'),
                    error: `Got error: ${e.message}`
                });
            }

            // UPDATE non-existent table
            try {
                await v.exec("UPDATE ghost_table SET name = 'test'");
                tests.push({ sql: 'UPDATE nonexistent table', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'UPDATE nonexistent table',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            // DELETE from non-existent table
            try {
                await v.exec('DELETE FROM phantom_table');
                tests.push({ sql: 'DELETE nonexistent table', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'DELETE nonexistent table',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            // DROP TABLE without IF EXISTS on non-existent
            try {
                await v.exec('DROP TABLE missing_table');
                tests.push({ sql: 'DROP nonexistent table', pass: false, error: 'Should have thrown' });
            } catch (e) {
                tests.push({
                    sql: 'DROP nonexistent table',
                    pass: e.message.length > 0,
                    error: `Got error: ${e.message}`
                });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });

    test('KV and SQL coexistence', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];
            const v = await vault();

            // Store KV data
            await v.set('app_config', { version: '1.0.0', debug: true });
            await v.set('user_prefs', { theme: 'dark', lang: 'en' });

            // Create SQL table
            await v.exec('DROP TABLE IF EXISTS kv_test_users');
            await v.exec('CREATE TABLE kv_test_users (id INTEGER, name TEXT)');
            await v.exec("INSERT INTO kv_test_users VALUES (1, 'Alice'), (2, 'Bob')");

            // Verify KV still works
            const config = await v.get('app_config');
            tests.push({
                sql: 'KV works after SQL',
                pass: config && config.version === '1.0.0',
                error: `Expected config, got ${JSON.stringify(config)}`
            });

            // Verify SQL still works
            const users = await v.exec('SELECT * FROM kv_test_users');
            tests.push({
                sql: 'SQL works after KV',
                pass: users.rows.length === 2,
                error: `Expected 2 users, got ${users.rows.length}`
            });

            // Update KV
            await v.set('app_config', { version: '2.0.0', debug: false });

            // Add more SQL data
            await v.exec("INSERT INTO kv_test_users VALUES (3, 'Charlie')");

            // Verify both updated correctly
            const newConfig = await v.get('app_config');
            const newUsers = await v.exec('SELECT * FROM kv_test_users');
            tests.push({
                sql: 'Both updated correctly',
                pass: newConfig.version === '2.0.0' && newUsers.rows.length === 3,
                error: `Config version: ${newConfig?.version}, Users: ${newUsers.rows.length}`
            });

            // Cleanup
            await v.delete('app_config');
            await v.delete('user_prefs');
            await v.exec('DROP TABLE kv_test_users');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.sql}: ${t.error || ''}`).toBe(true);
        }
    });
});
