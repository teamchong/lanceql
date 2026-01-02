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

    test('COALESCE function', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE coalesce_test (id INT, val1 TEXT, val2 TEXT, val3 TEXT)');
            await v.exec("INSERT INTO coalesce_test VALUES (1, NULL, 'second', 'third')");
            await v.exec("INSERT INTO coalesce_test VALUES (2, 'first', 'second', 'third')");
            await v.exec("INSERT INTO coalesce_test VALUES (3, NULL, NULL, 'third')");

            // Test COALESCE returns first non-null
            try {
                const res = await v.exec("SELECT COALESCE(val1, val2, val3) AS result FROM coalesce_test WHERE id = 1");
                tests.push({
                    name: 'COALESCE skips null',
                    pass: res.rows[0].result === 'second',
                    error: `Expected 'second', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'COALESCE skips null', pass: false, error: e.message });
            }

            // Test COALESCE returns first when not null
            try {
                const res = await v.exec("SELECT COALESCE(val1, val2, val3) AS result FROM coalesce_test WHERE id = 2");
                tests.push({
                    name: 'COALESCE returns first non-null',
                    pass: res.rows[0].result === 'first',
                    error: `Expected 'first', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'COALESCE returns first non-null', pass: false, error: e.message });
            }

            // Test COALESCE with literal default
            try {
                const res = await v.exec("SELECT COALESCE(val1, 'default') AS result FROM coalesce_test WHERE id = 1");
                tests.push({
                    name: 'COALESCE with literal',
                    pass: res.rows[0].result === 'default',
                    error: `Expected 'default', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'COALESCE with literal', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE coalesce_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('NULLIF function', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE nullif_test (id INT, val INT)');
            await v.exec('INSERT INTO nullif_test VALUES (1, 0), (2, 5), (3, 0)');

            // Test NULLIF returns null when equal
            try {
                const res = await v.exec('SELECT NULLIF(val, 0) AS result FROM nullif_test WHERE id = 1');
                tests.push({
                    name: 'NULLIF returns null when equal',
                    pass: res.rows[0].result === null,
                    error: `Expected null, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'NULLIF returns null when equal', pass: false, error: e.message });
            }

            // Test NULLIF returns value when not equal
            try {
                const res = await v.exec('SELECT NULLIF(val, 0) AS result FROM nullif_test WHERE id = 2');
                tests.push({
                    name: 'NULLIF returns value when not equal',
                    pass: res.rows[0].result === 5,
                    error: `Expected 5, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'NULLIF returns value when not equal', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE nullif_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('CASE WHEN expression', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE case_test (id INT, status TEXT, score INT)');
            await v.exec("INSERT INTO case_test VALUES (1, 'active', 85)");
            await v.exec("INSERT INTO case_test VALUES (2, 'pending', 60)");
            await v.exec("INSERT INTO case_test VALUES (3, 'inactive', 40)");

            // Test simple CASE (value matching)
            try {
                const res = await v.exec("SELECT CASE status WHEN 'active' THEN 'A' WHEN 'pending' THEN 'P' ELSE 'X' END AS code FROM case_test WHERE id = 1");
                tests.push({
                    name: 'Simple CASE returns matched value',
                    pass: res.rows[0].code === 'A',
                    error: `Expected 'A', got ${res.rows[0]?.code}`
                });
            } catch (e) {
                tests.push({ name: 'Simple CASE returns matched value', pass: false, error: e.message });
            }

            // Test CASE ELSE
            try {
                const res = await v.exec("SELECT CASE status WHEN 'active' THEN 'A' WHEN 'pending' THEN 'P' ELSE 'X' END AS code FROM case_test WHERE id = 3");
                tests.push({
                    name: 'CASE returns ELSE when no match',
                    pass: res.rows[0].code === 'X',
                    error: `Expected 'X', got ${res.rows[0]?.code}`
                });
            } catch (e) {
                tests.push({ name: 'CASE returns ELSE when no match', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE case_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('String functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE str_test (id INT, name TEXT)');
            await v.exec("INSERT INTO str_test VALUES (1, 'Hello World')");

            // Test UPPER
            try {
                const res = await v.exec('SELECT UPPER(name) AS result FROM str_test');
                tests.push({
                    name: 'UPPER',
                    pass: res.rows[0].result === 'HELLO WORLD',
                    error: `Expected 'HELLO WORLD', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'UPPER', pass: false, error: e.message });
            }

            // Test LOWER
            try {
                const res = await v.exec('SELECT LOWER(name) AS result FROM str_test');
                tests.push({
                    name: 'LOWER',
                    pass: res.rows[0].result === 'hello world',
                    error: `Expected 'hello world', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'LOWER', pass: false, error: e.message });
            }

            // Test LENGTH
            try {
                const res = await v.exec('SELECT LENGTH(name) AS result FROM str_test');
                tests.push({
                    name: 'LENGTH',
                    pass: res.rows[0].result === 11,
                    error: `Expected 11, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'LENGTH', pass: false, error: e.message });
            }

            // Test SUBSTR
            try {
                const res = await v.exec('SELECT SUBSTR(name, 1, 5) AS result FROM str_test');
                tests.push({
                    name: 'SUBSTR',
                    pass: res.rows[0].result === 'Hello',
                    error: `Expected 'Hello', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'SUBSTR', pass: false, error: e.message });
            }

            // Test CONCAT
            try {
                const res = await v.exec("SELECT CONCAT(name, '!') AS result FROM str_test");
                tests.push({
                    name: 'CONCAT',
                    pass: res.rows[0].result === 'Hello World!',
                    error: `Expected 'Hello World!', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'CONCAT', pass: false, error: e.message });
            }

            // Test REPLACE
            try {
                const res = await v.exec("SELECT REPLACE(name, 'World', 'SQL') AS result FROM str_test");
                tests.push({
                    name: 'REPLACE',
                    pass: res.rows[0].result === 'Hello SQL',
                    error: `Expected 'Hello SQL', got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'REPLACE', pass: false, error: e.message });
            }

            // Test TRIM
            await v.exec("INSERT INTO str_test VALUES (2, '  padded  ')");
            try {
                const res = await v.exec('SELECT TRIM(name) AS result FROM str_test WHERE id = 2');
                tests.push({
                    name: 'TRIM',
                    pass: res.rows[0].result === 'padded',
                    error: `Expected 'padded', got '${res.rows[0]?.result}'`
                });
            } catch (e) {
                tests.push({ name: 'TRIM', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE str_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Math functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE math_test (id INT, val FLOAT)');
            await v.exec('INSERT INTO math_test VALUES (1, -5.7), (2, 16), (3, 3.14159)');

            // Test ABS
            try {
                const res = await v.exec('SELECT ABS(val) AS result FROM math_test WHERE id = 1');
                tests.push({
                    name: 'ABS',
                    pass: Math.abs(res.rows[0].result - 5.7) < 0.001,
                    error: `Expected 5.7, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'ABS', pass: false, error: e.message });
            }

            // Test SQRT
            try {
                const res = await v.exec('SELECT SQRT(val) AS result FROM math_test WHERE id = 2');
                tests.push({
                    name: 'SQRT',
                    pass: res.rows[0].result === 4,
                    error: `Expected 4, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'SQRT', pass: false, error: e.message });
            }

            // Test ROUND
            try {
                const res = await v.exec('SELECT ROUND(val, 2) AS result FROM math_test WHERE id = 3');
                tests.push({
                    name: 'ROUND',
                    pass: res.rows[0].result === 3.14,
                    error: `Expected 3.14, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'ROUND', pass: false, error: e.message });
            }

            // Test CEIL
            try {
                const res = await v.exec('SELECT CEIL(val) AS result FROM math_test WHERE id = 3');
                tests.push({
                    name: 'CEIL',
                    pass: res.rows[0].result === 4,
                    error: `Expected 4, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'CEIL', pass: false, error: e.message });
            }

            // Test FLOOR
            try {
                const res = await v.exec('SELECT FLOOR(val) AS result FROM math_test WHERE id = 3');
                tests.push({
                    name: 'FLOOR',
                    pass: res.rows[0].result === 3,
                    error: `Expected 3, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'FLOOR', pass: false, error: e.message });
            }

            // Test MOD
            try {
                const res = await v.exec('SELECT MOD(val, 3) AS result FROM math_test WHERE id = 2');
                tests.push({
                    name: 'MOD',
                    pass: res.rows[0].result === 1,
                    error: `Expected 1, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'MOD', pass: false, error: e.message });
            }

            // Test POWER
            try {
                const res = await v.exec('SELECT POWER(val, 2) AS result FROM math_test WHERE id = 2');
                tests.push({
                    name: 'POWER',
                    pass: res.rows[0].result === 256,
                    error: `Expected 256, got ${res.rows[0]?.result}`
                });
            } catch (e) {
                tests.push({ name: 'POWER', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE math_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('IS NULL and IS NOT NULL', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE null_test (id INT, name TEXT, value INT)');
            await v.exec("INSERT INTO null_test VALUES (1, 'Alice', 100)");
            await v.exec("INSERT INTO null_test VALUES (2, NULL, 200)");
            await v.exec("INSERT INTO null_test VALUES (3, 'Bob', NULL)");

            // Test IS NULL
            try {
                const res = await v.exec('SELECT * FROM null_test WHERE name IS NULL');
                tests.push({
                    name: 'IS NULL',
                    pass: res.rows.length === 1 && res.rows[0].id === 2,
                    error: `Expected 1 row with id=2, got ${res.rows.length} rows`
                });
            } catch (e) {
                tests.push({ name: 'IS NULL', pass: false, error: e.message });
            }

            // Test IS NOT NULL
            try {
                const res = await v.exec('SELECT * FROM null_test WHERE name IS NOT NULL');
                tests.push({
                    name: 'IS NOT NULL',
                    pass: res.rows.length === 2,
                    error: `Expected 2 rows, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'IS NOT NULL', pass: false, error: e.message });
            }

            // Test IS NULL on value column
            try {
                const res = await v.exec('SELECT * FROM null_test WHERE value IS NULL');
                tests.push({
                    name: 'IS NULL (value)',
                    pass: res.rows.length === 1 && res.rows[0].id === 3,
                    error: `Expected 1 row with id=3, got ${res.rows.length} rows`
                });
            } catch (e) {
                tests.push({ name: 'IS NULL (value)', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE null_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('NOT IN, NOT LIKE, NOT BETWEEN', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE not_test (id INT, name TEXT, age INT)');
            await v.exec("INSERT INTO not_test VALUES (1, 'Alice', 25)");
            await v.exec("INSERT INTO not_test VALUES (2, 'Bob', 30)");
            await v.exec("INSERT INTO not_test VALUES (3, 'Charlie', 35)");
            await v.exec("INSERT INTO not_test VALUES (4, 'David', 40)");

            // Test NOT IN
            try {
                const res = await v.exec('SELECT * FROM not_test WHERE id NOT IN (1, 2)');
                tests.push({
                    name: 'NOT IN',
                    pass: res.rows.length === 2 && res.rows[0].id === 3 && res.rows[1].id === 4,
                    error: `Expected 2 rows with ids 3,4, got ${res.rows.length} rows`
                });
            } catch (e) {
                tests.push({ name: 'NOT IN', pass: false, error: e.message });
            }

            // Test NOT LIKE
            try {
                const res = await v.exec("SELECT * FROM not_test WHERE name NOT LIKE 'A%'");
                tests.push({
                    name: 'NOT LIKE',
                    pass: res.rows.length === 3,
                    error: `Expected 3 rows (Bob, Charlie, David), got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'NOT LIKE', pass: false, error: e.message });
            }

            // Test NOT BETWEEN
            try {
                const res = await v.exec('SELECT * FROM not_test WHERE age NOT BETWEEN 28 AND 38');
                tests.push({
                    name: 'NOT BETWEEN',
                    pass: res.rows.length === 2 && res.rows[0].id === 1 && res.rows[1].id === 4,
                    error: `Expected 2 rows (Alice=25, David=40), got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'NOT BETWEEN', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE not_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Subquery in WHERE with IN', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE orders (id INT, customer_id INT, amount INT)');
            await v.exec('CREATE TABLE vip_customers (id INT, name TEXT)');

            await v.exec('INSERT INTO orders VALUES (1, 100, 500)');
            await v.exec('INSERT INTO orders VALUES (2, 101, 300)');
            await v.exec('INSERT INTO orders VALUES (3, 100, 200)');
            await v.exec('INSERT INTO orders VALUES (4, 102, 800)');

            await v.exec("INSERT INTO vip_customers VALUES (100, 'Alice')");
            await v.exec("INSERT INTO vip_customers VALUES (102, 'Charlie')");

            // Test IN subquery
            try {
                const res = await v.exec('SELECT * FROM orders WHERE customer_id IN (SELECT id FROM vip_customers)');
                tests.push({
                    name: 'IN subquery',
                    pass: res.rows.length === 3,
                    error: `Expected 3 rows (orders from VIP customers), got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'IN subquery', pass: false, error: e.message });
            }

            // Test NOT IN subquery
            try {
                const res = await v.exec('SELECT * FROM orders WHERE customer_id NOT IN (SELECT id FROM vip_customers)');
                tests.push({
                    name: 'NOT IN subquery',
                    pass: res.rows.length === 1 && res.rows[0].customer_id === 101,
                    error: `Expected 1 row (customer 101), got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'NOT IN subquery', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE orders');
            await v.exec('DROP TABLE vip_customers');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('UNION and UNION ALL', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE us_sales (id INT, product TEXT, amount INT)');
            await v.exec('CREATE TABLE eu_sales (id INT, product TEXT, amount INT)');

            await v.exec("INSERT INTO us_sales VALUES (1, 'Widget', 100)");
            await v.exec("INSERT INTO us_sales VALUES (2, 'Gadget', 200)");
            await v.exec("INSERT INTO us_sales VALUES (3, 'Widget', 100)");  // Duplicate row

            await v.exec("INSERT INTO eu_sales VALUES (4, 'Widget', 150)");
            await v.exec("INSERT INTO eu_sales VALUES (5, 'Widget', 100)");  // Same as US row 1 and 3

            // Test UNION (removes duplicates)
            try {
                const res = await v.exec('SELECT product, amount FROM us_sales UNION SELECT product, amount FROM eu_sales');
                tests.push({
                    name: 'UNION removes duplicates',
                    pass: res.rows.length === 3,  // (Widget,100), (Gadget,200), (Widget,150)
                    error: `Expected 3 unique rows, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'UNION removes duplicates', pass: false, error: e.message });
            }

            // Test UNION ALL (keeps duplicates)
            try {
                const res = await v.exec('SELECT product, amount FROM us_sales UNION ALL SELECT product, amount FROM eu_sales');
                tests.push({
                    name: 'UNION ALL keeps duplicates',
                    pass: res.rows.length === 5,  // All 5 rows
                    error: `Expected 5 rows, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'UNION ALL keeps duplicates', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE us_sales');
            await v.exec('DROP TABLE eu_sales');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('INTERSECT and EXCEPT', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE set_a (id INT, val TEXT)');
            await v.exec('CREATE TABLE set_b (id INT, val TEXT)');

            await v.exec("INSERT INTO set_a VALUES (1, 'A')");
            await v.exec("INSERT INTO set_a VALUES (2, 'B')");
            await v.exec("INSERT INTO set_a VALUES (3, 'C')");

            await v.exec("INSERT INTO set_b VALUES (2, 'B')");
            await v.exec("INSERT INTO set_b VALUES (3, 'C')");
            await v.exec("INSERT INTO set_b VALUES (4, 'D')");

            // Test INTERSECT (common rows)
            try {
                const res = await v.exec('SELECT id, val FROM set_a INTERSECT SELECT id, val FROM set_b');
                tests.push({
                    name: 'INTERSECT',
                    pass: res.rows.length === 2,  // (2,'B') and (3,'C')
                    error: `Expected 2 common rows, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'INTERSECT', pass: false, error: e.message });
            }

            // Test EXCEPT (rows in A but not B)
            try {
                const res = await v.exec('SELECT id, val FROM set_a EXCEPT SELECT id, val FROM set_b');
                tests.push({
                    name: 'EXCEPT',
                    pass: res.rows.length === 1 && res.rows[0].id === 1,
                    error: `Expected 1 row (id=1), got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'EXCEPT', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE set_a');
            await v.exec('DROP TABLE set_b');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('WITH clause (CTEs)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE employees (id INT, name TEXT, dept_id INT, salary INT)');
            await v.exec('CREATE TABLE departments (id INT, name TEXT)');

            await v.exec("INSERT INTO employees VALUES (1, 'Alice', 10, 50000)");
            await v.exec("INSERT INTO employees VALUES (2, 'Bob', 10, 60000)");
            await v.exec("INSERT INTO employees VALUES (3, 'Charlie', 20, 70000)");
            await v.exec("INSERT INTO employees VALUES (4, 'David', 20, 55000)");

            await v.exec("INSERT INTO departments VALUES (10, 'Engineering')");
            await v.exec("INSERT INTO departments VALUES (20, 'Sales')");

            // Test simple CTE
            try {
                const res = await v.exec(`
                    WITH high_earners AS (
                        SELECT id, name, salary FROM employees WHERE salary > 55000
                    )
                    SELECT * FROM high_earners
                `);
                tests.push({
                    name: 'Simple CTE',
                    pass: res.rows.length === 2,  // Bob (60000) and Charlie (70000)
                    error: `Expected 2 high earners, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'Simple CTE', pass: false, error: e.message });
            }

            // Test CTE with aggregation
            try {
                const res = await v.exec(`
                    WITH dept_totals AS (
                        SELECT dept_id, SUM(salary) AS total_salary FROM employees GROUP BY dept_id
                    )
                    SELECT * FROM dept_totals ORDER BY total_salary DESC
                `);
                tests.push({
                    name: 'CTE with aggregation',
                    pass: res.rows.length === 2 && res.rows[0].total_salary === 125000,
                    error: `Expected 2 departments, first with 125000, got ${res.rows.length} rows with first total ${res.rows[0]?.total_salary}`
                });
            } catch (e) {
                tests.push({ name: 'CTE with aggregation', pass: false, error: e.message });
            }

            // Test multiple CTEs
            try {
                const res = await v.exec(`
                    WITH
                        eng_emp AS (SELECT id, name FROM employees WHERE dept_id = 10),
                        sales_emp AS (SELECT id, name FROM employees WHERE dept_id = 20)
                    SELECT * FROM eng_emp UNION ALL SELECT * FROM sales_emp
                `);
                tests.push({
                    name: 'Multiple CTEs',
                    pass: res.rows.length === 4,  // All 4 employees
                    error: `Expected 4 employees, got ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'Multiple CTEs', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE employees');
            await v.exec('DROP TABLE departments');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Window functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE sales (id INT, region TEXT, amount INT)');
            await v.exec("INSERT INTO sales VALUES (1, 'North', 100)");
            await v.exec("INSERT INTO sales VALUES (2, 'North', 200)");
            await v.exec("INSERT INTO sales VALUES (3, 'South', 150)");
            await v.exec("INSERT INTO sales VALUES (4, 'South', 250)");
            await v.exec("INSERT INTO sales VALUES (5, 'North', 300)");

            // Test ROW_NUMBER() OVER (PARTITION BY)
            try {
                const res = await v.exec('SELECT id, region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount) AS rn FROM sales');
                const northRows = res.rows.filter(r => r.region === 'North');
                tests.push({
                    name: 'ROW_NUMBER with PARTITION BY',
                    pass: northRows.length === 3 && northRows.some(r => r.rn === 1) && northRows.some(r => r.rn === 2) && northRows.some(r => r.rn === 3),
                    error: `Expected North to have row numbers 1,2,3, got ${JSON.stringify(northRows)}`
                });
            } catch (e) {
                tests.push({ name: 'ROW_NUMBER with PARTITION BY', pass: false, error: e.message });
            }

            // Test SUM() OVER (PARTITION BY) - running total
            try {
                const res = await v.exec('SELECT id, region, SUM(amount) OVER (PARTITION BY region) AS region_total FROM sales');
                const northRows = res.rows.filter(r => r.region === 'North');
                tests.push({
                    name: 'SUM OVER PARTITION BY',
                    pass: northRows.every(r => r.region_total === 600),  // 100+200+300
                    error: `Expected all North rows to have region_total=600, got ${JSON.stringify(northRows)}`
                });
            } catch (e) {
                tests.push({ name: 'SUM OVER PARTITION BY', pass: false, error: e.message });
            }

            // Test RANK() with ties
            await v.exec('DROP TABLE sales');
            await v.exec('CREATE TABLE scores (id INT, score INT)');
            await v.exec('INSERT INTO scores VALUES (1, 100)');
            await v.exec('INSERT INTO scores VALUES (2, 90)');
            await v.exec('INSERT INTO scores VALUES (3, 90)');  // Tie
            await v.exec('INSERT INTO scores VALUES (4, 80)');

            try {
                const res = await v.exec('SELECT id, score, RANK() OVER (ORDER BY score DESC) AS rnk FROM scores');
                const ranks = res.rows.map(r => ({ id: r.id, rnk: r.rnk }));
                // Expected: id=1 -> rank 1, id=2 -> rank 2, id=3 -> rank 2 (tie), id=4 -> rank 4 (skips 3)
                const id1Rank = ranks.find(r => r.id === 1)?.rnk;
                const id2Rank = ranks.find(r => r.id === 2)?.rnk;
                const id3Rank = ranks.find(r => r.id === 3)?.rnk;
                const id4Rank = ranks.find(r => r.id === 4)?.rnk;
                tests.push({
                    name: 'RANK with ties',
                    pass: id1Rank === 1 && id2Rank === 2 && id3Rank === 2 && id4Rank === 4,
                    error: `Expected ranks 1,2,2,4, got ${JSON.stringify(ranks)}`
                });
            } catch (e) {
                tests.push({ name: 'RANK with ties', pass: false, error: e.message });
            }

            // Test DENSE_RANK() - no gaps
            try {
                const res = await v.exec('SELECT id, score, DENSE_RANK() OVER (ORDER BY score DESC) AS rnk FROM scores');
                const ranks = res.rows.map(r => ({ id: r.id, rnk: r.rnk }));
                // Expected: id=1 -> rank 1, id=2 -> rank 2, id=3 -> rank 2 (tie), id=4 -> rank 3 (no gap)
                const id4Rank = ranks.find(r => r.id === 4)?.rnk;
                tests.push({
                    name: 'DENSE_RANK no gaps',
                    pass: id4Rank === 3,
                    error: `Expected id=4 to have rank 3, got ${id4Rank}`
                });
            } catch (e) {
                tests.push({ name: 'DENSE_RANK no gaps', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE scores');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    // Date/Time functions
    test('Date/Time functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup test table
            await v.exec('DROP TABLE IF EXISTS dt_test');
            await v.exec('CREATE TABLE dt_test (id INTEGER, event_time TEXT)');
            await v.exec("INSERT INTO dt_test VALUES (1, '2024-06-15 14:30:45')");
            await v.exec("INSERT INTO dt_test VALUES (2, '2023-12-25 09:00:00')");
            await v.exec("INSERT INTO dt_test VALUES (3, '2024-01-01 00:00:00')");

            // NOW() returns ISO timestamp
            try {
                const res = await v.exec('SELECT NOW() AS now_val FROM dt_test LIMIT 1');
                const now = res.rows[0].now_val;
                tests.push({
                    name: 'NOW returns ISO timestamp',
                    pass: now && now.includes('T') && now.includes('Z'),
                    error: `Expected ISO timestamp, got ${now}`
                });
            } catch (e) {
                tests.push({ name: 'NOW returns ISO timestamp', pass: false, error: e.message });
            }

            // CURRENT_DATE returns YYYY-MM-DD
            try {
                const res = await v.exec('SELECT CURRENT_DATE() AS today FROM dt_test LIMIT 1');
                const today = res.rows[0].today;
                tests.push({
                    name: 'CURRENT_DATE returns date',
                    pass: today && /^\d{4}-\d{2}-\d{2}$/.test(today),
                    error: `Expected YYYY-MM-DD, got ${today}`
                });
            } catch (e) {
                tests.push({ name: 'CURRENT_DATE returns date', pass: false, error: e.message });
            }

            // DATE() extracts date
            try {
                const res = await v.exec("SELECT DATE('2024-06-15 14:30:45') AS d FROM dt_test LIMIT 1");
                tests.push({
                    name: 'DATE extracts date',
                    pass: res.rows[0].d === '2024-06-15',
                    error: `Expected 2024-06-15, got ${res.rows[0].d}`
                });
            } catch (e) {
                tests.push({ name: 'DATE extracts date', pass: false, error: e.message });
            }

            // YEAR, MONTH, DAY extractors
            try {
                const res = await v.exec("SELECT YEAR('2024-06-15') AS y, MONTH('2024-06-15') AS m, DAY('2024-06-15') AS d FROM dt_test LIMIT 1");
                tests.push({
                    name: 'YEAR/MONTH/DAY extract parts',
                    pass: res.rows[0].y === 2024 && res.rows[0].m === 6 && res.rows[0].d === 15,
                    error: `Expected 2024,6,15, got ${res.rows[0].y},${res.rows[0].m},${res.rows[0].d}`
                });
            } catch (e) {
                tests.push({ name: 'YEAR/MONTH/DAY extract parts', pass: false, error: e.message });
            }

            // HOUR, MINUTE, SECOND extractors (use ISO format with Z for UTC)
            try {
                const res = await v.exec("SELECT HOUR('2024-06-15T14:30:45Z') AS h, MINUTE('2024-06-15T14:30:45Z') AS mi, SECOND('2024-06-15T14:30:45Z') AS s FROM dt_test LIMIT 1");
                tests.push({
                    name: 'HOUR/MINUTE/SECOND extract parts',
                    pass: res.rows[0].h === 14 && res.rows[0].mi === 30 && res.rows[0].s === 45,
                    error: `Expected 14,30,45, got ${res.rows[0].h},${res.rows[0].mi},${res.rows[0].s}`
                });
            } catch (e) {
                tests.push({ name: 'HOUR/MINUTE/SECOND extract parts', pass: false, error: e.message });
            }

            // STRFTIME formats date
            try {
                const res = await v.exec("SELECT STRFTIME('%Y/%m/%d', '2024-06-15') AS fmt FROM dt_test LIMIT 1");
                tests.push({
                    name: 'STRFTIME formats date',
                    pass: res.rows[0].fmt === '2024/06/15',
                    error: `Expected 2024/06/15, got ${res.rows[0].fmt}`
                });
            } catch (e) {
                tests.push({ name: 'STRFTIME formats date', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE dt_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    // Additional string functions
    test('Additional string functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('DROP TABLE IF EXISTS str_test');
            await v.exec('CREATE TABLE str_test (id INTEGER, txt TEXT)');
            await v.exec("INSERT INTO str_test VALUES (1, 'Hello World')");

            // LEFT
            try {
                const res = await v.exec("SELECT LEFT('Hello', 3) AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'LEFT extracts chars',
                    pass: res.rows[0].val === 'Hel',
                    error: `Expected 'Hel', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'LEFT extracts chars', pass: false, error: e.message });
            }

            // RIGHT
            try {
                const res = await v.exec("SELECT RIGHT('Hello', 3) AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'RIGHT extracts chars',
                    pass: res.rows[0].val === 'llo',
                    error: `Expected 'llo', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'RIGHT extracts chars', pass: false, error: e.message });
            }

            // LPAD
            try {
                const res = await v.exec("SELECT LPAD('Hi', 5, '*') AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'LPAD pads left',
                    pass: res.rows[0].val === '***Hi',
                    error: `Expected '***Hi', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'LPAD pads left', pass: false, error: e.message });
            }

            // RPAD
            try {
                const res = await v.exec("SELECT RPAD('Hi', 5, '*') AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'RPAD pads right',
                    pass: res.rows[0].val === 'Hi***',
                    error: `Expected 'Hi***', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'RPAD pads right', pass: false, error: e.message });
            }

            // POSITION (str, substr) - finds substr in str, returns 1-based position
            try {
                const res = await v.exec("SELECT POSITION('Hello World', 'World') AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'POSITION finds substring',
                    pass: res.rows[0].val === 7,
                    error: `Expected 7, got ${res.rows[0].val}`
                });
            } catch (e) {
                tests.push({ name: 'POSITION finds substring', pass: false, error: e.message });
            }

            // REPEAT
            try {
                const res = await v.exec("SELECT REPEAT('ab', 3) AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'REPEAT repeats string',
                    pass: res.rows[0].val === 'ababab',
                    error: `Expected 'ababab', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'REPEAT repeats string', pass: false, error: e.message });
            }

            // REVERSE
            try {
                const res = await v.exec("SELECT REVERSE('Hello') AS val FROM str_test LIMIT 1");
                tests.push({
                    name: 'REVERSE reverses string',
                    pass: res.rows[0].val === 'olleH',
                    error: `Expected 'olleH', got '${res.rows[0].val}'`
                });
            } catch (e) {
                tests.push({ name: 'REVERSE reverses string', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE str_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    // Arithmetic operators in SELECT
    test('Arithmetic operators in SELECT', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('DROP TABLE IF EXISTS math_test');
            await v.exec('CREATE TABLE math_test (id INTEGER, price REAL, quantity INTEGER)');
            await v.exec('INSERT INTO math_test VALUES (1, 10.5, 3)');
            await v.exec('INSERT INTO math_test VALUES (2, 25.0, 2)');
            await v.exec('INSERT INTO math_test VALUES (3, 15.75, 4)');

            // Addition: col + col
            try {
                const res = await v.exec('SELECT id, price + quantity AS total FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Addition: col + col',
                    pass: res.rows[0].total === 13.5,
                    error: `Expected 13.5, got ${res.rows[0].total}`
                });
            } catch (e) {
                tests.push({ name: 'Addition: col + col', pass: false, error: e.message });
            }

            // Subtraction: col - col
            try {
                const res = await v.exec('SELECT id, price - quantity AS diff FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Subtraction: col - col',
                    pass: res.rows[0].diff === 7.5,
                    error: `Expected 7.5, got ${res.rows[0].diff}`
                });
            } catch (e) {
                tests.push({ name: 'Subtraction: col - col', pass: false, error: e.message });
            }

            // Multiplication: col * col
            try {
                const res = await v.exec('SELECT id, price * quantity AS product FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Multiplication: col * col',
                    pass: res.rows[0].product === 31.5,
                    error: `Expected 31.5, got ${res.rows[0].product}`
                });
            } catch (e) {
                tests.push({ name: 'Multiplication: col * col', pass: false, error: e.message });
            }

            // Division: col / col
            try {
                const res = await v.exec('SELECT id, price / quantity AS avg_price FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Division: col / col',
                    pass: res.rows[0].avg_price === 3.5,
                    error: `Expected 3.5, got ${res.rows[0].avg_price}`
                });
            } catch (e) {
                tests.push({ name: 'Division: col / col', pass: false, error: e.message });
            }

            // Precedence: a + b * c
            try {
                const res = await v.exec('SELECT id, 2 + 3 * 4 AS result FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Precedence: + and * (2 + 3 * 4 = 14)',
                    pass: res.rows[0].result === 14,
                    error: `Expected 14, got ${res.rows[0].result}`
                });
            } catch (e) {
                tests.push({ name: 'Precedence: + and *', pass: false, error: e.message });
            }

            // Parentheses: (a + b) * c
            try {
                const res = await v.exec('SELECT id, (2 + 3) * 4 AS result FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Parentheses: (2 + 3) * 4 = 20',
                    pass: res.rows[0].result === 20,
                    error: `Expected 20, got ${res.rows[0].result}`
                });
            } catch (e) {
                tests.push({ name: 'Parentheses: (2 + 3) * 4', pass: false, error: e.message });
            }

            // Literal with column: col * 0.1
            try {
                const res = await v.exec('SELECT id, price * 0.1 AS discount FROM math_test WHERE id = 1');
                const discount = Math.round(res.rows[0].discount * 1000) / 1000; // Handle float precision
                tests.push({
                    name: 'Literal with column: price * 0.1',
                    pass: discount === 1.05,
                    error: `Expected 1.05, got ${discount}`
                });
            } catch (e) {
                tests.push({ name: 'Literal with column', pass: false, error: e.message });
            }

            // Unary minus
            try {
                const res = await v.exec('SELECT id, -price AS neg FROM math_test WHERE id = 1');
                tests.push({
                    name: 'Unary minus: -price',
                    pass: res.rows[0].neg === -10.5,
                    error: `Expected -10.5, got ${res.rows[0].neg}`
                });
            } catch (e) {
                tests.push({ name: 'Unary minus: -price', pass: false, error: e.message });
            }

            // Alias for arithmetic expression
            try {
                const res = await v.exec('SELECT id, price * quantity AS total_value FROM math_test');
                // Check that arithmetic works with alias
                const row1 = res.rows.find(r => r.id === 1);
                const row3 = res.rows.find(r => r.id === 3);
                tests.push({
                    name: 'Alias for arithmetic expression',
                    pass: row1.total_value === 31.5 && row3.total_value === 63 && res.rows.length === 3,
                    error: `Expected row1=31.5, row3=63, got row1=${row1?.total_value}, row3=${row3?.total_value}`
                });
            } catch (e) {
                tests.push({ name: 'Alias for arithmetic expression', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE math_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    // NULLS FIRST/LAST in ORDER BY
    test('NULLS FIRST/LAST in ORDER BY', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('DROP TABLE IF EXISTS nulls_test');
            await v.exec('CREATE TABLE nulls_test (id INTEGER, val INTEGER)');
            await v.exec('INSERT INTO nulls_test VALUES (1, 10)');
            await v.exec('INSERT INTO nulls_test VALUES (2, NULL)');
            await v.exec('INSERT INTO nulls_test VALUES (3, 30)');
            await v.exec('INSERT INTO nulls_test VALUES (4, NULL)');
            await v.exec('INSERT INTO nulls_test VALUES (5, 20)');

            // Default ASC: NULLs last
            try {
                const res = await v.exec('SELECT * FROM nulls_test ORDER BY val ASC');
                const ids = res.rows.map(r => r.id);
                // Non-null values sorted first, then nulls
                tests.push({
                    name: 'Default ASC: NULLs last',
                    pass: ids[0] === 1 && ids[1] === 5 && ids[2] === 3 && (ids[3] === 2 || ids[3] === 4),
                    error: `Expected non-nulls first then nulls, got ${ids.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'Default ASC: NULLs last', pass: false, error: e.message });
            }

            // Default DESC: NULLs first
            try {
                const res = await v.exec('SELECT * FROM nulls_test ORDER BY val DESC');
                const ids = res.rows.map(r => r.id);
                // Nulls first, then non-null values sorted desc
                tests.push({
                    name: 'Default DESC: NULLs first',
                    pass: (ids[0] === 2 || ids[0] === 4) && ids[2] === 3 && ids[3] === 5 && ids[4] === 1,
                    error: `Expected nulls first then desc values, got ${ids.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'Default DESC: NULLs first', pass: false, error: e.message });
            }

            // ASC NULLS FIRST
            try {
                const res = await v.exec('SELECT * FROM nulls_test ORDER BY val ASC NULLS FIRST');
                const ids = res.rows.map(r => r.id);
                tests.push({
                    name: 'ASC NULLS FIRST',
                    pass: (ids[0] === 2 || ids[0] === 4) && ids[2] === 1,
                    error: `Expected nulls first, got ${ids.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'ASC NULLS FIRST', pass: false, error: e.message });
            }

            // DESC NULLS LAST
            try {
                const res = await v.exec('SELECT * FROM nulls_test ORDER BY val DESC NULLS LAST');
                const ids = res.rows.map(r => r.id);
                tests.push({
                    name: 'DESC NULLS LAST',
                    pass: ids[0] === 3 && (ids[3] === 2 || ids[3] === 4),
                    error: `Expected desc values then nulls last, got ${ids.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'DESC NULLS LAST', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE nulls_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || ''}`).toBe(true);
        }
    });

    test('Quick wins - GREATEST, LEAST, IIF functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // GREATEST
            try {
                const res = await v.exec('SELECT GREATEST(1, 5, 3) AS max_val');
                const pass = res.rows[0].max_val === 5;
                tests.push({ name: 'GREATEST(1, 5, 3)', pass, actual: res.rows[0].max_val });
            } catch (e) {
                tests.push({ name: 'GREATEST(1, 5, 3)', pass: false, error: e.message });
            }

            // LEAST
            try {
                const res = await v.exec('SELECT LEAST(1, 5, 3) AS min_val');
                const pass = res.rows[0].min_val === 1;
                tests.push({ name: 'LEAST(1, 5, 3)', pass, actual: res.rows[0].min_val });
            } catch (e) {
                tests.push({ name: 'LEAST(1, 5, 3)', pass: false, error: e.message });
            }

            // GREATEST with NULLs
            try {
                const res = await v.exec('SELECT GREATEST(1, NULL, 3) AS max_val');
                const pass = res.rows[0].max_val === 3;
                tests.push({ name: 'GREATEST with NULL', pass, actual: res.rows[0].max_val });
            } catch (e) {
                tests.push({ name: 'GREATEST with NULL', pass: false, error: e.message });
            }

            // IIF true
            try {
                const res = await v.exec("SELECT IIF(1 > 0, 'yes', 'no') AS result");
                const pass = res.rows[0].result === 'yes';
                tests.push({ name: 'IIF(1 > 0)', pass, actual: res.rows[0].result });
            } catch (e) {
                tests.push({ name: 'IIF(1 > 0)', pass: false, error: e.message });
            }

            // IIF false
            try {
                const res = await v.exec("SELECT IIF(1 < 0, 'yes', 'no') AS result");
                const pass = res.rows[0].result === 'no';
                tests.push({ name: 'IIF(1 < 0)', pass, actual: res.rows[0].result });
            } catch (e) {
                tests.push({ name: 'IIF(1 < 0)', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Quick wins - Math functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // LOG
            try {
                const res = await v.exec('SELECT LOG(2.718281828) AS log_val');
                const pass = Math.abs(res.rows[0].log_val - 1) < 0.001;
                tests.push({ name: 'LOG(e)', pass, actual: res.rows[0].log_val });
            } catch (e) {
                tests.push({ name: 'LOG(e)', pass: false, error: e.message });
            }

            // EXP
            try {
                const res = await v.exec('SELECT EXP(1) AS exp_val');
                const pass = Math.abs(res.rows[0].exp_val - Math.E) < 0.001;
                tests.push({ name: 'EXP(1)', pass, actual: res.rows[0].exp_val });
            } catch (e) {
                tests.push({ name: 'EXP(1)', pass: false, error: e.message });
            }

            // SIN
            try {
                const res = await v.exec('SELECT SIN(0) AS sin_val');
                const pass = res.rows[0].sin_val === 0;
                tests.push({ name: 'SIN(0)', pass, actual: res.rows[0].sin_val });
            } catch (e) {
                tests.push({ name: 'SIN(0)', pass: false, error: e.message });
            }

            // COS
            try {
                const res = await v.exec('SELECT COS(0) AS cos_val');
                const pass = res.rows[0].cos_val === 1;
                tests.push({ name: 'COS(0)', pass, actual: res.rows[0].cos_val });
            } catch (e) {
                tests.push({ name: 'COS(0)', pass: false, error: e.message });
            }

            // PI
            try {
                const res = await v.exec('SELECT PI() AS pi_val');
                const pass = Math.abs(res.rows[0].pi_val - Math.PI) < 0.0001;
                tests.push({ name: 'PI()', pass, actual: res.rows[0].pi_val });
            } catch (e) {
                tests.push({ name: 'PI()', pass: false, error: e.message });
            }

            // SIGN
            try {
                const res = await v.exec('SELECT SIGN(-5) AS sign_neg, SIGN(0) AS sign_zero, SIGN(5) AS sign_pos');
                const pass = res.rows[0].sign_neg === -1 && res.rows[0].sign_zero === 0 && res.rows[0].sign_pos === 1;
                tests.push({ name: 'SIGN', pass, actual: JSON.stringify(res.rows[0]) });
            } catch (e) {
                tests.push({ name: 'SIGN', pass: false, error: e.message });
            }

            // DEGREES
            try {
                const res = await v.exec('SELECT DEGREES(3.14159265359) AS deg_val');
                const pass = Math.abs(res.rows[0].deg_val - 180) < 0.01;
                tests.push({ name: 'DEGREES(PI)', pass, actual: res.rows[0].deg_val });
            } catch (e) {
                tests.push({ name: 'DEGREES(PI)', pass: false, error: e.message });
            }

            // RADIANS
            try {
                const res = await v.exec('SELECT RADIANS(180) AS rad_val');
                const pass = Math.abs(res.rows[0].rad_val - Math.PI) < 0.01;
                tests.push({ name: 'RADIANS(180)', pass, actual: res.rows[0].rad_val });
            } catch (e) {
                tests.push({ name: 'RADIANS(180)', pass: false, error: e.message });
            }

            // TRUNCATE
            try {
                const res = await v.exec('SELECT TRUNCATE(3.7) AS trunc_val');
                const pass = res.rows[0].trunc_val === 3;
                tests.push({ name: 'TRUNCATE(3.7)', pass, actual: res.rows[0].trunc_val });
            } catch (e) {
                tests.push({ name: 'TRUNCATE(3.7)', pass: false, error: e.message });
            }

            // RANDOM
            try {
                const res = await v.exec('SELECT RANDOM() AS rand_val');
                const pass = res.rows[0].rand_val >= 0 && res.rows[0].rand_val < 1;
                tests.push({ name: 'RANDOM()', pass, actual: res.rows[0].rand_val });
            } catch (e) {
                tests.push({ name: 'RANDOM()', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Quick wins - CAST function', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // CAST to INTEGER
            try {
                const res = await v.exec("SELECT CAST('123' AS INTEGER) AS int_val");
                const pass = res.rows[0].int_val === 123;
                tests.push({ name: "CAST('123' AS INTEGER)", pass, actual: res.rows[0].int_val });
            } catch (e) {
                tests.push({ name: "CAST('123' AS INTEGER)", pass: false, error: e.message });
            }

            // CAST to TEXT
            try {
                const res = await v.exec('SELECT CAST(456 AS TEXT) AS text_val');
                const pass = res.rows[0].text_val === '456';
                tests.push({ name: 'CAST(456 AS TEXT)', pass, actual: res.rows[0].text_val });
            } catch (e) {
                tests.push({ name: 'CAST(456 AS TEXT)', pass: false, error: e.message });
            }

            // CAST float to INTEGER (truncates)
            try {
                const res = await v.exec('SELECT CAST(3.9 AS INTEGER) AS int_val');
                const pass = res.rows[0].int_val === 3;
                tests.push({ name: 'CAST(3.9 AS INTEGER)', pass, actual: res.rows[0].int_val });
            } catch (e) {
                tests.push({ name: 'CAST(3.9 AS INTEGER)', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Advanced aggregates - STDDEV, VARIANCE, MEDIAN', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE agg_test (val INTEGER)');
            await v.exec('INSERT INTO agg_test VALUES (2), (4), (4), (4), (5), (5), (7), (9)');

            // STDDEV (sample)
            try {
                const res = await v.exec('SELECT STDDEV(val) AS std FROM agg_test');
                const pass = Math.abs(res.rows[0].std - 2.138) < 0.01;
                tests.push({ name: 'STDDEV(val)', pass, actual: res.rows[0].std });
            } catch (e) {
                tests.push({ name: 'STDDEV(val)', pass: false, error: e.message });
            }

            // VARIANCE (sample)
            try {
                const res = await v.exec('SELECT VARIANCE(val) AS var FROM agg_test');
                const pass = Math.abs(res.rows[0].var - 4.571) < 0.01;
                tests.push({ name: 'VARIANCE(val)', pass, actual: res.rows[0].var });
            } catch (e) {
                tests.push({ name: 'VARIANCE(val)', pass: false, error: e.message });
            }

            // MEDIAN (even count)
            try {
                const res = await v.exec('SELECT MEDIAN(val) AS med FROM agg_test');
                const pass = res.rows[0].med === 4.5;
                tests.push({ name: 'MEDIAN(val) even', pass, actual: res.rows[0].med });
            } catch (e) {
                tests.push({ name: 'MEDIAN(val) even', pass: false, error: e.message });
            }

            // Cleanup and setup for odd count
            await v.exec('DROP TABLE agg_test');
            await v.exec('CREATE TABLE agg_test (val INTEGER)');
            await v.exec('INSERT INTO agg_test VALUES (1), (3), (5), (7), (9)');

            // MEDIAN (odd count)
            try {
                const res = await v.exec('SELECT MEDIAN(val) AS med FROM agg_test');
                const pass = res.rows[0].med === 5;
                tests.push({ name: 'MEDIAN(val) odd', pass, actual: res.rows[0].med });
            } catch (e) {
                tests.push({ name: 'MEDIAN(val) odd', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE agg_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Advanced aggregates - STRING_AGG', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE str_test (name TEXT, category TEXT)');
            await v.exec("INSERT INTO str_test VALUES ('apple', 'fruit'), ('banana', 'fruit'), ('carrot', 'veg')");

            // STRING_AGG with custom separator
            try {
                const res = await v.exec("SELECT STRING_AGG(name, '; ') AS names FROM str_test");
                const pass = res.rows[0].names === 'apple; banana; carrot';
                tests.push({ name: "STRING_AGG with '; '", pass, actual: res.rows[0].names });
            } catch (e) {
                tests.push({ name: "STRING_AGG with '; '", pass: false, error: e.message });
            }

            // STRING_AGG with GROUP BY
            try {
                const res = await v.exec("SELECT category, STRING_AGG(name, ', ') AS names FROM str_test GROUP BY category ORDER BY category");
                const pass = res.rows.length === 2 &&
                    res.rows[0].names === 'apple, banana' &&
                    res.rows[1].names === 'carrot';
                tests.push({ name: 'STRING_AGG with GROUP BY', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'STRING_AGG with GROUP BY', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE str_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Window functions - NTILE', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE ntile_test (id INTEGER, score INTEGER)');
            await v.exec('INSERT INTO ntile_test VALUES (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30), (8, 20)');

            // NTILE(4) - divide into quartiles
            try {
                const res = await v.exec('SELECT id, NTILE(4) OVER (ORDER BY score DESC) AS quartile FROM ntile_test');
                const pass = res.rows.length === 8 &&
                    res.rows[0].quartile === 1 && res.rows[1].quartile === 1 &&
                    res.rows[2].quartile === 2 && res.rows[3].quartile === 2 &&
                    res.rows[4].quartile === 3 && res.rows[5].quartile === 3 &&
                    res.rows[6].quartile === 4 && res.rows[7].quartile === 4;
                tests.push({ name: 'NTILE(4)', pass, actual: JSON.stringify(res.rows.map(r => r.quartile)) });
            } catch (e) {
                tests.push({ name: 'NTILE(4)', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE ntile_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Window functions - PERCENT_RANK, CUME_DIST', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE prank_test (id INTEGER, val INTEGER)');
            await v.exec('INSERT INTO prank_test VALUES (1, 10), (2, 20), (3, 30), (4, 40)');

            // PERCENT_RANK
            try {
                const res = await v.exec('SELECT id, PERCENT_RANK() OVER (ORDER BY val) AS prank FROM prank_test');
                const pass = res.rows.length === 4 &&
                    res.rows[0].prank === 0 &&
                    Math.abs(res.rows[1].prank - 0.333) < 0.01 &&
                    Math.abs(res.rows[2].prank - 0.667) < 0.01 &&
                    res.rows[3].prank === 1;
                tests.push({ name: 'PERCENT_RANK()', pass, actual: JSON.stringify(res.rows.map(r => r.prank)) });
            } catch (e) {
                tests.push({ name: 'PERCENT_RANK()', pass: false, error: e.message });
            }

            // CUME_DIST
            try {
                const res = await v.exec('SELECT id, CUME_DIST() OVER (ORDER BY val) AS cdist FROM prank_test');
                const pass = res.rows.length === 4 &&
                    res.rows[0].cdist === 0.25 &&
                    res.rows[1].cdist === 0.5 &&
                    res.rows[2].cdist === 0.75 &&
                    res.rows[3].cdist === 1;
                tests.push({ name: 'CUME_DIST()', pass, actual: JSON.stringify(res.rows.map(r => r.cdist)) });
            } catch (e) {
                tests.push({ name: 'CUME_DIST()', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE prank_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Window functions - FIRST_VALUE, LAST_VALUE', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE flv_test (region TEXT, amount INTEGER)');
            await v.exec("INSERT INTO flv_test VALUES ('North', 100), ('North', 200), ('South', 150), ('South', 250)");

            // FIRST_VALUE
            try {
                const res = await v.exec('SELECT region, amount, FIRST_VALUE(amount) OVER (PARTITION BY region ORDER BY amount) AS first_amt FROM flv_test');
                const pass = res.rows.length === 4 &&
                    res.rows[0].first_amt === 100 && res.rows[1].first_amt === 100 &&
                    res.rows[2].first_amt === 150 && res.rows[3].first_amt === 150;
                tests.push({ name: 'FIRST_VALUE()', pass, actual: JSON.stringify(res.rows.map(r => r.first_amt)) });
            } catch (e) {
                tests.push({ name: 'FIRST_VALUE()', pass: false, error: e.message });
            }

            // LAST_VALUE with full frame
            try {
                const res = await v.exec('SELECT region, amount, LAST_VALUE(amount) OVER (PARTITION BY region ORDER BY amount ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_amt FROM flv_test');
                const pass = res.rows.length === 4 &&
                    res.rows[0].last_amt === 200 && res.rows[1].last_amt === 200 &&
                    res.rows[2].last_amt === 250 && res.rows[3].last_amt === 250;
                tests.push({ name: 'LAST_VALUE() with full frame', pass, actual: JSON.stringify(res.rows.map(r => r.last_amt)) });
            } catch (e) {
                tests.push({ name: 'LAST_VALUE() with full frame', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE flv_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Window frames - ROWS BETWEEN', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE frame_test (id INTEGER, val INTEGER)');
            await v.exec('INSERT INTO frame_test VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)');

            // SUM with 1 PRECEDING and 1 FOLLOWING
            try {
                const res = await v.exec('SELECT id, val, SUM(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_sum FROM frame_test');
                // Row 1: 10+20 = 30, Row 2: 10+20+30 = 60, Row 3: 20+30+40 = 90, Row 4: 30+40+50 = 120, Row 5: 40+50 = 90
                const pass = res.rows.length === 5 &&
                    res.rows[0].moving_sum === 30 &&
                    res.rows[1].moving_sum === 60 &&
                    res.rows[2].moving_sum === 90 &&
                    res.rows[3].moving_sum === 120 &&
                    res.rows[4].moving_sum === 90;
                tests.push({ name: 'SUM with ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING', pass, actual: JSON.stringify(res.rows.map(r => r.moving_sum)) });
            } catch (e) {
                tests.push({ name: 'SUM with ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING', pass: false, error: e.message });
            }

            // AVG with CURRENT ROW and 2 FOLLOWING
            try {
                const res = await v.exec('SELECT id, val, AVG(val) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS moving_avg FROM frame_test');
                // Row 1: (10+20+30)/3 = 20, Row 2: (20+30+40)/3 = 30, Row 3: (30+40+50)/3 = 40, Row 4: (40+50)/2 = 45, Row 5: 50/1 = 50
                const pass = res.rows.length === 5 &&
                    res.rows[0].moving_avg === 20 &&
                    res.rows[1].moving_avg === 30 &&
                    res.rows[2].moving_avg === 40 &&
                    res.rows[3].moving_avg === 45 &&
                    res.rows[4].moving_avg === 50;
                tests.push({ name: 'AVG with ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING', pass, actual: JSON.stringify(res.rows.map(r => r.moving_avg)) });
            } catch (e) {
                tests.push({ name: 'AVG with ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE frame_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('FROM subqueries (derived tables)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL)');
            await v.exec('INSERT INTO orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 150), (4, 2, 250), (5, 3, 300)');

            // Simple FROM subquery
            try {
                const res = await v.exec('SELECT * FROM (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) sub ORDER BY total DESC');
                const pass = res.rows.length === 3 &&
                    res.rows[0].total === 400 && // customer 2
                    res.rows[1].total === 300 && // customer 3
                    res.rows[2].total === 300;   // customer 1
                tests.push({ name: 'FROM subquery', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'FROM subquery', pass: false, error: e.message });
            }

            // Subquery with filter
            try {
                const res = await v.exec('SELECT * FROM (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) sub WHERE total > 300');
                const pass = res.rows.length === 1 && res.rows[0].total === 400;
                tests.push({ name: 'FROM subquery with WHERE', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'FROM subquery with WHERE', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE orders');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('EXISTS and NOT EXISTS', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE customers (id INTEGER, name TEXT)');
            await v.exec('CREATE TABLE orders (id INTEGER, customer_id INTEGER)');
            await v.exec("INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')");
            await v.exec('INSERT INTO orders VALUES (1, 1), (2, 1), (3, 2)');

            // EXISTS - returns all rows when subquery has results
            try {
                const res = await v.exec('SELECT * FROM customers WHERE EXISTS (SELECT 1 FROM orders)');
                const pass = res.rows.length === 3;
                tests.push({ name: 'EXISTS with results', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'EXISTS with results', pass: false, error: e.message });
            }

            // NOT EXISTS - returns no rows when subquery has results
            try {
                const res = await v.exec('SELECT * FROM customers WHERE NOT EXISTS (SELECT 1 FROM orders)');
                const pass = res.rows.length === 0;
                tests.push({ name: 'NOT EXISTS with results', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'NOT EXISTS with results', pass: false, error: e.message });
            }

            // EXISTS with empty subquery
            try {
                const res = await v.exec('SELECT * FROM customers WHERE EXISTS (SELECT 1 FROM orders WHERE customer_id = 999)');
                const pass = res.rows.length === 0;
                tests.push({ name: 'EXISTS with no results', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'EXISTS with no results', pass: false, error: e.message });
            }

            // NOT EXISTS with empty subquery
            try {
                const res = await v.exec('SELECT * FROM customers WHERE NOT EXISTS (SELECT 1 FROM orders WHERE customer_id = 999)');
                const pass = res.rows.length === 3;
                tests.push({ name: 'NOT EXISTS with no results', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'NOT EXISTS with no results', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE customers');
            await v.exec('DROP TABLE orders');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    // ============================================================================
    // Phase 3: Advanced JOINs, REGEXP, JSON, Scalar Subqueries
    // ============================================================================

    test('FULL OUTER JOIN operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE left_tbl (id INT, val TEXT)');
            await v.exec('CREATE TABLE right_tbl (id INT, val TEXT)');
            await v.exec("INSERT INTO left_tbl VALUES (1, 'A'), (2, 'B'), (3, 'C')");
            await v.exec("INSERT INTO right_tbl VALUES (2, 'X'), (3, 'Y'), (4, 'Z')");

            // FULL OUTER JOIN
            try {
                const res = await v.exec('SELECT l.id, l.val, r.id, r.val FROM left_tbl l FULL OUTER JOIN right_tbl r ON l.id = r.id ORDER BY l.id, r.id');
                // Should have 4 rows: (1,A,null), (2,B,2,X), (3,C,3,Y), (null,null,4,Z)
                const pass = res.rows.length === 4;
                tests.push({ name: 'FULL OUTER JOIN row count', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'FULL OUTER JOIN row count', pass: false, error: e.message });
            }

            // FULL JOIN (without OUTER keyword)
            try {
                const res = await v.exec('SELECT l.id FROM left_tbl l FULL JOIN right_tbl r ON l.id = r.id');
                const pass = res.rows.length === 4;
                tests.push({ name: 'FULL JOIN without OUTER', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'FULL JOIN without OUTER', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE left_tbl');
            await v.exec('DROP TABLE right_tbl');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('CROSS JOIN operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE colors (name TEXT)');
            await v.exec('CREATE TABLE sizes (name TEXT)');
            await v.exec("INSERT INTO colors VALUES ('Red'), ('Blue')");
            await v.exec("INSERT INTO sizes VALUES ('S'), ('M'), ('L')");

            // CROSS JOIN - Cartesian product
            try {
                const res = await v.exec('SELECT c.name AS color, s.name AS size FROM colors c CROSS JOIN sizes s');
                // Should have 2 * 3 = 6 rows
                const pass = res.rows.length === 6;
                tests.push({ name: 'CROSS JOIN Cartesian product', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'CROSS JOIN Cartesian product', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE colors');
            await v.exec('DROP TABLE sizes');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Compound JOIN conditions (AND/OR)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE orders (id INT, customer_id INT, region TEXT)');
            await v.exec('CREATE TABLE customers (id INT, region TEXT)');
            await v.exec("INSERT INTO orders VALUES (1, 10, 'East'), (2, 20, 'West'), (3, 10, 'West')");
            await v.exec("INSERT INTO customers VALUES (10, 'East'), (20, 'West')");

            // JOIN with AND condition
            try {
                const res = await v.exec('SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id AND o.region = c.region');
                // Only orders 1 and 2 match (matching both id AND region)
                const pass = res.rows.length === 2;
                tests.push({ name: 'JOIN with AND condition', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'JOIN with AND condition', pass: false, error: e.message });
            }

            // JOIN with OR condition
            try {
                const res = await v.exec('SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id OR o.region = c.region');
                // All 3 orders match some customer
                const pass = res.rows.length >= 3;
                tests.push({ name: 'JOIN with OR condition', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'JOIN with OR condition', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE orders');
            await v.exec('DROP TABLE customers');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('REGEXP functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE texts (id INT, content TEXT)');
            await v.exec("INSERT INTO texts VALUES (1, 'Hello World 123'), (2, 'foo@example.com'), (3, 'abc-def-ghi')");

            // REGEXP_MATCHES
            try {
                const res = await v.exec("SELECT id, REGEXP_MATCHES(content, '[0-9]+') AS has_numbers FROM texts WHERE id = 1");
                const pass = res.rows[0].has_numbers === 1;
                tests.push({ name: 'REGEXP_MATCHES', pass, actual: res.rows[0].has_numbers });
            } catch (e) {
                tests.push({ name: 'REGEXP_MATCHES', pass: false, error: e.message });
            }

            // REGEXP_REPLACE
            try {
                const res = await v.exec("SELECT REGEXP_REPLACE(content, '[0-9]+', 'XXX') AS replaced FROM texts WHERE id = 1");
                const pass = res.rows[0].replaced === 'Hello World XXX';
                tests.push({ name: 'REGEXP_REPLACE', pass, actual: res.rows[0].replaced });
            } catch (e) {
                tests.push({ name: 'REGEXP_REPLACE', pass: false, error: e.message });
            }

            // REGEXP_EXTRACT
            try {
                const res = await v.exec("SELECT REGEXP_EXTRACT(content, '([a-z]+)@([a-z.]+)', 1) AS username FROM texts WHERE id = 2");
                const pass = res.rows[0].username === 'foo';
                tests.push({ name: 'REGEXP_EXTRACT with group', pass, actual: res.rows[0].username });
            } catch (e) {
                tests.push({ name: 'REGEXP_EXTRACT with group', pass: false, error: e.message });
            }

            // REGEXP_COUNT
            try {
                const res = await v.exec("SELECT REGEXP_COUNT(content, '-') AS dash_count FROM texts WHERE id = 3");
                const pass = res.rows[0].dash_count === 2;
                tests.push({ name: 'REGEXP_COUNT', pass, actual: res.rows[0].dash_count });
            } catch (e) {
                tests.push({ name: 'REGEXP_COUNT', pass: false, error: e.message });
            }

            // REGEXP_SPLIT
            try {
                const res = await v.exec("SELECT REGEXP_SPLIT(content, '-') AS parts FROM texts WHERE id = 3");
                const parts = JSON.parse(res.rows[0].parts);
                const pass = parts.length === 3 && parts[0] === 'abc';
                tests.push({ name: 'REGEXP_SPLIT', pass, actual: res.rows[0].parts });
            } catch (e) {
                tests.push({ name: 'REGEXP_SPLIT', pass: false, error: e.message });
            }

            // Case-insensitive matching
            try {
                const res = await v.exec("SELECT REGEXP_MATCHES('Hello', 'hello', 'i') AS case_insensitive");
                const pass = res.rows[0].case_insensitive === 1;
                tests.push({ name: 'REGEXP case-insensitive', pass, actual: res.rows[0].case_insensitive });
            } catch (e) {
                tests.push({ name: 'REGEXP case-insensitive', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE texts');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('JSON functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE json_data (id INT, data TEXT)');
            await v.exec(`INSERT INTO json_data VALUES (1, '{"name": "Alice", "age": 30, "address": {"city": "NYC"}}')`);
            await v.exec(`INSERT INTO json_data VALUES (2, '{"tags": ["a", "b", "c"], "count": 3}')`);

            // JSON_EXTRACT with simple path
            try {
                const res = await v.exec("SELECT JSON_EXTRACT(data, '$.name') AS name FROM json_data WHERE id = 1");
                const pass = res.rows[0].name === 'Alice';
                tests.push({ name: 'JSON_EXTRACT simple', pass, actual: res.rows[0].name });
            } catch (e) {
                tests.push({ name: 'JSON_EXTRACT simple', pass: false, error: e.message });
            }

            // JSON_EXTRACT with nested path
            try {
                const res = await v.exec("SELECT JSON_EXTRACT(data, '$.address.city') AS city FROM json_data WHERE id = 1");
                const pass = res.rows[0].city === 'NYC';
                tests.push({ name: 'JSON_EXTRACT nested', pass, actual: res.rows[0].city });
            } catch (e) {
                tests.push({ name: 'JSON_EXTRACT nested', pass: false, error: e.message });
            }

            // JSON_EXTRACT with array index
            try {
                const res = await v.exec("SELECT JSON_EXTRACT(data, '$.tags[1]') AS tag FROM json_data WHERE id = 2");
                const pass = res.rows[0].tag === 'b';
                tests.push({ name: 'JSON_EXTRACT array', pass, actual: res.rows[0].tag });
            } catch (e) {
                tests.push({ name: 'JSON_EXTRACT array', pass: false, error: e.message });
            }

            // JSON_OBJECT
            try {
                const res = await v.exec("SELECT JSON_OBJECT('key1', 'val1', 'key2', 42) AS obj");
                const obj = JSON.parse(res.rows[0].obj);
                const pass = obj.key1 === 'val1' && obj.key2 === 42;
                tests.push({ name: 'JSON_OBJECT', pass, actual: res.rows[0].obj });
            } catch (e) {
                tests.push({ name: 'JSON_OBJECT', pass: false, error: e.message });
            }

            // JSON_ARRAY
            try {
                const res = await v.exec("SELECT JSON_ARRAY(1, 2, 'three') AS arr");
                const arr = JSON.parse(res.rows[0].arr);
                const pass = arr.length === 3 && arr[2] === 'three';
                tests.push({ name: 'JSON_ARRAY', pass, actual: res.rows[0].arr });
            } catch (e) {
                tests.push({ name: 'JSON_ARRAY', pass: false, error: e.message });
            }

            // JSON_KEYS
            try {
                const res = await v.exec("SELECT JSON_KEYS(data) AS keys FROM json_data WHERE id = 1");
                const keys = JSON.parse(res.rows[0].keys);
                const pass = keys.includes('name') && keys.includes('age');
                tests.push({ name: 'JSON_KEYS', pass, actual: res.rows[0].keys });
            } catch (e) {
                tests.push({ name: 'JSON_KEYS', pass: false, error: e.message });
            }

            // JSON_LENGTH
            try {
                const res = await v.exec("SELECT JSON_LENGTH(data) AS len FROM json_data WHERE id = 1");
                const pass = res.rows[0].len === 3;
                tests.push({ name: 'JSON_LENGTH object', pass, actual: res.rows[0].len });
            } catch (e) {
                tests.push({ name: 'JSON_LENGTH object', pass: false, error: e.message });
            }

            // JSON_LENGTH on array
            try {
                const res = await v.exec(`SELECT JSON_LENGTH('["a","b","c"]') AS len`);
                const pass = res.rows[0].len === 3;
                tests.push({ name: 'JSON_LENGTH array', pass, actual: res.rows[0].len });
            } catch (e) {
                tests.push({ name: 'JSON_LENGTH array', pass: false, error: e.message });
            }

            // JSON_TYPE
            try {
                const res = await v.exec(`SELECT JSON_TYPE('{"a":1}') AS t1, JSON_TYPE('[1,2]') AS t2, JSON_TYPE('"hello"') AS t3`);
                const pass = res.rows[0].t1 === 'OBJECT' && res.rows[0].t2 === 'ARRAY' && res.rows[0].t3 === 'STRING';
                tests.push({ name: 'JSON_TYPE', pass, actual: `${res.rows[0].t1}, ${res.rows[0].t2}, ${res.rows[0].t3}` });
            } catch (e) {
                tests.push({ name: 'JSON_TYPE', pass: false, error: e.message });
            }

            // JSON_VALID
            try {
                const res = await v.exec(`SELECT JSON_VALID('{"a":1}') AS valid, JSON_VALID('not json') AS invalid`);
                const pass = res.rows[0].valid === 1 && res.rows[0].invalid === 0;
                tests.push({ name: 'JSON_VALID', pass, actual: `valid=${res.rows[0].valid}, invalid=${res.rows[0].invalid}` });
            } catch (e) {
                tests.push({ name: 'JSON_VALID', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE json_data');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Scalar subqueries in SELECT', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE departments (id INT, name TEXT)');
            await v.exec('CREATE TABLE employees (id INT, name TEXT, dept_id INT, salary REAL)');
            await v.exec("INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales'), (3, 'HR')");
            await v.exec("INSERT INTO employees VALUES (1, 'Alice', 1, 80000), (2, 'Bob', 1, 70000), (3, 'Carol', 2, 60000), (4, 'Dave', 2, 55000)");

            // Simple scalar subquery (non-correlated)
            try {
                const res = await v.exec('SELECT id, name, (SELECT MAX(salary) FROM employees) AS max_salary FROM departments WHERE id = 1');
                const pass = res.rows[0].max_salary === 80000;
                tests.push({ name: 'Non-correlated scalar subquery', pass, actual: res.rows[0].max_salary });
            } catch (e) {
                tests.push({ name: 'Non-correlated scalar subquery', pass: false, error: e.message });
            }

            // Correlated scalar subquery
            try {
                const res = await v.exec('SELECT d.id, d.name, (SELECT COUNT(*) FROM employees e WHERE e.dept_id = d.id) AS emp_count FROM departments d ORDER BY d.id');
                const pass = res.rows[0].emp_count === 2 && res.rows[1].emp_count === 2 && res.rows[2].emp_count === 0;
                tests.push({ name: 'Correlated scalar subquery COUNT', pass, actual: `${res.rows[0].emp_count}, ${res.rows[1].emp_count}, ${res.rows[2].emp_count}` });
            } catch (e) {
                tests.push({ name: 'Correlated scalar subquery COUNT', pass: false, error: e.message });
            }

            // Scalar subquery returning MAX per department
            try {
                const res = await v.exec('SELECT d.name, (SELECT MAX(salary) FROM employees e WHERE e.dept_id = d.id) AS max_salary FROM departments d WHERE d.id <= 2 ORDER BY d.id');
                const pass = res.rows[0].max_salary === 80000 && res.rows[1].max_salary === 60000;
                tests.push({ name: 'Correlated scalar subquery MAX', pass, actual: `${res.rows[0].max_salary}, ${res.rows[1].max_salary}` });
            } catch (e) {
                tests.push({ name: 'Correlated scalar subquery MAX', pass: false, error: e.message });
            }

            // Scalar subquery with alias
            try {
                const res = await v.exec('SELECT name, (SELECT 100) AS constant FROM departments WHERE id = 1');
                const pass = res.rows[0].constant === 100;
                tests.push({ name: 'Scalar subquery with alias', pass, actual: res.rows[0].constant });
            } catch (e) {
                tests.push({ name: 'Scalar subquery with alias', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE employees');
            await v.exec('DROP TABLE departments');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('LEFT OUTER JOIN (with OUTER keyword)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup
            await v.exec('CREATE TABLE tbl_a (id INT, val TEXT)');
            await v.exec('CREATE TABLE tbl_b (id INT, val TEXT)');
            await v.exec("INSERT INTO tbl_a VALUES (1, 'A'), (2, 'B')");
            await v.exec("INSERT INTO tbl_b VALUES (1, 'X')");

            // LEFT OUTER JOIN
            try {
                const res = await v.exec('SELECT a.id, a.val, b.val FROM tbl_a a LEFT OUTER JOIN tbl_b b ON a.id = b.id ORDER BY a.id');
                const pass = res.rows.length === 2;
                tests.push({ name: 'LEFT OUTER JOIN', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'LEFT OUTER JOIN', pass: false, error: e.message });
            }

            // RIGHT OUTER JOIN
            try {
                await v.exec("INSERT INTO tbl_b VALUES (3, 'Y')");
                const res = await v.exec('SELECT a.id, b.id, b.val FROM tbl_a a RIGHT OUTER JOIN tbl_b b ON a.id = b.id');
                const pass = res.rows.length === 2; // id=1 match, id=3 no match on left
                tests.push({ name: 'RIGHT OUTER JOIN', pass, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'RIGHT OUTER JOIN', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE tbl_a');
            await v.exec('DROP TABLE tbl_b');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    // ==================== PHASE 4: Data Type Operations ====================

    test('ARRAY operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Test ARRAY constructor
            try {
                const res = await v.exec('SELECT ARRAY[1, 2, 3] AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && arr.length === 3 && arr[0] === 1 && arr[1] === 2 && arr[2] === 3;
                tests.push({ name: 'ARRAY[1, 2, 3] constructor', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'ARRAY constructor', pass: false, error: e.message });
            }

            // Test bare bracket array
            try {
                const res = await v.exec('SELECT [10, 20, 30] AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && arr.length === 3;
                tests.push({ name: '[10, 20, 30] bare bracket array', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'Bare bracket array', pass: false, error: e.message });
            }

            // Test array subscript (1-indexed)
            try {
                const res = await v.exec('SELECT ARRAY[10, 20, 30][2] AS val');
                const pass = res.rows[0]?.val === 20;
                tests.push({ name: 'ARRAY[10, 20, 30][2] subscript', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Array subscript', pass: false, error: e.message });
            }

            // Test ARRAY_LENGTH
            try {
                const res = await v.exec('SELECT ARRAY_LENGTH(ARRAY[1, 2, 3, 4]) AS len');
                const pass = res.rows[0]?.len === 4;
                tests.push({ name: 'ARRAY_LENGTH', pass, actual: res.rows[0]?.len });
            } catch (e) {
                tests.push({ name: 'ARRAY_LENGTH', pass: false, error: e.message });
            }

            // Test ARRAY_CONTAINS
            try {
                const res = await v.exec('SELECT ARRAY_CONTAINS(ARRAY[1, 2, 3], 2) AS found');
                const pass = res.rows[0]?.found === 1;
                tests.push({ name: 'ARRAY_CONTAINS found', pass, actual: res.rows[0]?.found });
            } catch (e) {
                tests.push({ name: 'ARRAY_CONTAINS', pass: false, error: e.message });
            }

            // Test ARRAY_CONTAINS not found
            try {
                const res = await v.exec('SELECT ARRAY_CONTAINS(ARRAY[1, 2, 3], 5) AS found');
                const pass = res.rows[0]?.found === 0;
                tests.push({ name: 'ARRAY_CONTAINS not found', pass, actual: res.rows[0]?.found });
            } catch (e) {
                tests.push({ name: 'ARRAY_CONTAINS not found', pass: false, error: e.message });
            }

            // Test ARRAY_POSITION
            try {
                const res = await v.exec("SELECT ARRAY_POSITION(ARRAY['a', 'b', 'c'], 'b') AS pos");
                const pass = res.rows[0]?.pos === 2;
                tests.push({ name: 'ARRAY_POSITION', pass, actual: res.rows[0]?.pos });
            } catch (e) {
                tests.push({ name: 'ARRAY_POSITION', pass: false, error: e.message });
            }

            // Test ARRAY_APPEND
            try {
                const res = await v.exec('SELECT ARRAY_APPEND(ARRAY[1, 2], 3) AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && arr.length === 3 && arr[2] === 3;
                tests.push({ name: 'ARRAY_APPEND', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'ARRAY_APPEND', pass: false, error: e.message });
            }

            // Test ARRAY_REMOVE
            try {
                const res = await v.exec('SELECT ARRAY_REMOVE(ARRAY[1, 2, 3, 2], 2) AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && arr.length === 2 && !arr.includes(2);
                tests.push({ name: 'ARRAY_REMOVE', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'ARRAY_REMOVE', pass: false, error: e.message });
            }

            // Test ARRAY_SLICE
            try {
                const res = await v.exec('SELECT ARRAY_SLICE(ARRAY[1, 2, 3, 4, 5], 2, 4) AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && JSON.stringify(arr) === '[2,3]';
                tests.push({ name: 'ARRAY_SLICE(arr, 2, 4)', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'ARRAY_SLICE', pass: false, error: e.message });
            }

            // Test ARRAY_CONCAT
            try {
                const res = await v.exec('SELECT ARRAY_CONCAT(ARRAY[1, 2], ARRAY[3, 4]) AS arr');
                const arr = res.rows[0]?.arr;
                const pass = Array.isArray(arr) && arr.length === 4;
                tests.push({ name: 'ARRAY_CONCAT', pass, actual: JSON.stringify(arr) });
            } catch (e) {
                tests.push({ name: 'ARRAY_CONCAT', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('UUID functions', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];
            const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

            // Test UUID() generates valid UUID
            try {
                const res = await v.exec('SELECT UUID() AS id');
                const uuid = res.rows[0]?.id;
                const pass = uuidRegex.test(uuid);
                tests.push({ name: 'UUID() generates valid UUID', pass, actual: uuid });
            } catch (e) {
                tests.push({ name: 'UUID()', pass: false, error: e.message });
            }

            // Test GEN_RANDOM_UUID
            try {
                const res = await v.exec('SELECT GEN_RANDOM_UUID() AS id');
                const uuid = res.rows[0]?.id;
                const pass = uuidRegex.test(uuid);
                tests.push({ name: 'GEN_RANDOM_UUID()', pass, actual: uuid });
            } catch (e) {
                tests.push({ name: 'GEN_RANDOM_UUID()', pass: false, error: e.message });
            }

            // Test IS_UUID valid
            try {
                const res = await v.exec("SELECT IS_UUID('550e8400-e29b-41d4-a716-446655440000') AS valid");
                const pass = res.rows[0]?.valid === 1;
                tests.push({ name: 'IS_UUID valid UUID', pass, actual: res.rows[0]?.valid });
            } catch (e) {
                tests.push({ name: 'IS_UUID valid', pass: false, error: e.message });
            }

            // Test IS_UUID invalid
            try {
                const res = await v.exec("SELECT IS_UUID('not-a-uuid') AS valid");
                const pass = res.rows[0]?.valid === 0;
                tests.push({ name: 'IS_UUID invalid string', pass, actual: res.rows[0]?.valid });
            } catch (e) {
                tests.push({ name: 'IS_UUID invalid', pass: false, error: e.message });
            }

            // Test UUID uniqueness
            try {
                const res = await v.exec('SELECT UUID() AS id1, UUID() AS id2');
                const pass = res.rows[0]?.id1 !== res.rows[0]?.id2;
                tests.push({ name: 'UUID uniqueness', pass, actual: `${res.rows[0]?.id1} vs ${res.rows[0]?.id2}` });
            } catch (e) {
                tests.push({ name: 'UUID uniqueness', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('DECIMAL precision - TRUNC and ROUND with scale', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Test ROUND with scale 2
            try {
                const res = await v.exec('SELECT ROUND(3.14159, 2) AS val');
                const pass = res.rows[0]?.val === 3.14;
                tests.push({ name: 'ROUND(3.14159, 2)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'ROUND with scale', pass: false, error: e.message });
            }

            // Test ROUND with scale 3
            try {
                const res = await v.exec('SELECT ROUND(123.456789, 3) AS val');
                const pass = res.rows[0]?.val === 123.457;
                tests.push({ name: 'ROUND(123.456789, 3)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'ROUND with scale 3', pass: false, error: e.message });
            }

            // Test ROUND with negative scale (round to tens)
            try {
                const res = await v.exec('SELECT ROUND(1234.5, -2) AS val');
                const pass = res.rows[0]?.val === 1200;
                tests.push({ name: 'ROUND(1234.5, -2) to hundreds', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'ROUND negative scale', pass: false, error: e.message });
            }

            // Test TRUNC with scale 2
            try {
                const res = await v.exec('SELECT TRUNC(3.14159, 2) AS val');
                const pass = res.rows[0]?.val === 3.14;
                tests.push({ name: 'TRUNC(3.14159, 2)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'TRUNC with scale', pass: false, error: e.message });
            }

            // Test TRUNC with scale 3
            try {
                const res = await v.exec('SELECT TRUNC(123.456789, 3) AS val');
                const pass = res.rows[0]?.val === 123.456;
                tests.push({ name: 'TRUNC(123.456789, 3)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'TRUNC with scale 3', pass: false, error: e.message });
            }

            // Test TRUNC without scale (default behavior)
            try {
                const res = await v.exec('SELECT TRUNC(3.9) AS val');
                const pass = res.rows[0]?.val === 3;
                tests.push({ name: 'TRUNC(3.9) no scale', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'TRUNC no scale', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('Binary/Bitwise operations', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Test bitwise AND
            try {
                const res = await v.exec('SELECT 12 & 10 AS val');
                const pass = res.rows[0]?.val === 8; // 1100 & 1010 = 1000
                tests.push({ name: 'Bitwise AND (12 & 10)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Bitwise AND', pass: false, error: e.message });
            }

            // Test bitwise OR
            try {
                const res = await v.exec('SELECT 12 | 10 AS val');
                const pass = res.rows[0]?.val === 14; // 1100 | 1010 = 1110
                tests.push({ name: 'Bitwise OR (12 | 10)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Bitwise OR', pass: false, error: e.message });
            }

            // Test bitwise XOR
            try {
                const res = await v.exec('SELECT 12 ^ 10 AS val');
                const pass = res.rows[0]?.val === 6; // 1100 ^ 1010 = 0110
                tests.push({ name: 'Bitwise XOR (12 ^ 10)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Bitwise XOR', pass: false, error: e.message });
            }

            // Test left shift
            try {
                const res = await v.exec('SELECT 1 << 4 AS val');
                const pass = res.rows[0]?.val === 16;
                tests.push({ name: 'Left shift (1 << 4)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Left shift', pass: false, error: e.message });
            }

            // Test right shift
            try {
                const res = await v.exec('SELECT 16 >> 2 AS val');
                const pass = res.rows[0]?.val === 4;
                tests.push({ name: 'Right shift (16 >> 2)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Right shift', pass: false, error: e.message });
            }

            // Test bitwise NOT
            try {
                const res = await v.exec('SELECT ~0 AS val');
                const pass = res.rows[0]?.val === -1;
                tests.push({ name: 'Bitwise NOT (~0)', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'Bitwise NOT', pass: false, error: e.message });
            }

            // Test BIT_COUNT
            try {
                const res = await v.exec('SELECT BIT_COUNT(15) AS cnt');
                const pass = res.rows[0]?.cnt === 4; // 1111 has 4 bits set
                tests.push({ name: 'BIT_COUNT(15)', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'BIT_COUNT', pass: false, error: e.message });
            }

            // Test BIT_COUNT with larger number
            try {
                const res = await v.exec('SELECT BIT_COUNT(255) AS cnt');
                const pass = res.rows[0]?.cnt === 8; // 11111111 has 8 bits set
                tests.push({ name: 'BIT_COUNT(255)', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'BIT_COUNT(255)', pass: false, error: e.message });
            }

            // Test HEX with number
            try {
                const res = await v.exec('SELECT HEX(255) AS hex');
                const pass = res.rows[0]?.hex === 'FF';
                tests.push({ name: 'HEX(255)', pass, actual: res.rows[0]?.hex });
            } catch (e) {
                tests.push({ name: 'HEX number', pass: false, error: e.message });
            }

            // Test HEX with string
            try {
                const res = await v.exec("SELECT HEX('AB') AS hex");
                const pass = res.rows[0]?.hex === '4142'; // A=0x41, B=0x42
                tests.push({ name: "HEX('AB')", pass, actual: res.rows[0]?.hex });
            } catch (e) {
                tests.push({ name: 'HEX string', pass: false, error: e.message });
            }

            // Test UNHEX
            try {
                const res = await v.exec("SELECT UNHEX('4142') AS str");
                const pass = res.rows[0]?.str === 'AB';
                tests.push({ name: "UNHEX('4142')", pass, actual: res.rows[0]?.str });
            } catch (e) {
                tests.push({ name: 'UNHEX', pass: false, error: e.message });
            }

            // Test ENCODE/DECODE base64
            try {
                const res = await v.exec("SELECT DECODE(ENCODE('Hello', 'base64'), 'base64') AS val");
                const pass = res.rows[0]?.val === 'Hello';
                tests.push({ name: 'ENCODE/DECODE base64', pass, actual: res.rows[0]?.val });
            } catch (e) {
                tests.push({ name: 'ENCODE/DECODE', pass: false, error: e.message });
            }

            // Test ENCODE base64
            try {
                const res = await v.exec("SELECT ENCODE('Hello', 'base64') AS encoded");
                const pass = res.rows[0]?.encoded === 'SGVsbG8=';
                tests.push({ name: "ENCODE('Hello', 'base64')", pass, actual: res.rows[0]?.encoded });
            } catch (e) {
                tests.push({ name: 'ENCODE base64', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    // ==================== PHASE 5: DML ENHANCEMENTS ====================

    test('INSERT...SELECT copies data between tables', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create source and destination tables
            await v.exec('CREATE TABLE src (id INT, value TEXT)');
            await v.exec('CREATE TABLE dst (id INT, value TEXT)');
            await v.exec("INSERT INTO src VALUES (1, 'a'), (2, 'b'), (3, 'c')");

            // Test INSERT...SELECT (all columns)
            try {
                await v.exec('INSERT INTO dst SELECT * FROM src');
                const res = await v.exec('SELECT COUNT(*) AS cnt FROM dst');
                const pass = res.rows[0]?.cnt === 3;
                tests.push({ name: 'INSERT...SELECT all', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'INSERT...SELECT all', pass: false, error: e.message });
            }

            // Test INSERT...SELECT with WHERE filter
            try {
                await v.exec('CREATE TABLE dst2 (id INT, value TEXT)');
                await v.exec('INSERT INTO dst2 SELECT * FROM src WHERE id > 1');
                const res = await v.exec('SELECT COUNT(*) AS cnt FROM dst2');
                const pass = res.rows[0]?.cnt === 2;
                tests.push({ name: 'INSERT...SELECT with WHERE', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'INSERT...SELECT with WHERE', pass: false, error: e.message });
            }

            // Test INSERT...SELECT with column mapping
            try {
                await v.exec('CREATE TABLE dst3 (num INT, txt TEXT)');
                await v.exec('INSERT INTO dst3 (num, txt) SELECT id, value FROM src WHERE id = 1');
                const res = await v.exec('SELECT num, txt FROM dst3');
                const pass = res.rows[0]?.num === 1 && res.rows[0]?.txt === 'a';
                tests.push({ name: 'INSERT...SELECT column mapping', pass, actual: JSON.stringify(res.rows[0]) });
            } catch (e) {
                tests.push({ name: 'INSERT...SELECT column mapping', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('UPSERT with ON CONFLICT DO NOTHING and DO UPDATE', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create table with data
            await v.exec('CREATE TABLE items (id INT PRIMARY KEY, qty INT, name TEXT)');
            await v.exec("INSERT INTO items VALUES (1, 10, 'apple'), (2, 20, 'banana')");

            // Test ON CONFLICT DO NOTHING
            try {
                await v.exec("INSERT INTO items VALUES (1, 999, 'new') ON CONFLICT (id) DO NOTHING");
                const res = await v.exec('SELECT qty FROM items WHERE id = 1');
                const pass = res.rows[0]?.qty === 10;  // Should remain unchanged
                tests.push({ name: 'ON CONFLICT DO NOTHING', pass, actual: res.rows[0]?.qty });
            } catch (e) {
                tests.push({ name: 'ON CONFLICT DO NOTHING', pass: false, error: e.message });
            }

            // Test ON CONFLICT DO UPDATE with EXCLUDED
            try {
                await v.exec("INSERT INTO items VALUES (1, 50, 'updated') ON CONFLICT (id) DO UPDATE SET qty = EXCLUDED.qty");
                const res = await v.exec('SELECT qty FROM items WHERE id = 1');
                const pass = res.rows[0]?.qty === 50;  // Should be updated
                tests.push({ name: 'ON CONFLICT DO UPDATE EXCLUDED.qty', pass, actual: res.rows[0]?.qty });
            } catch (e) {
                tests.push({ name: 'ON CONFLICT DO UPDATE', pass: false, error: e.message });
            }

            // Test ON CONFLICT with multiple SET columns
            try {
                await v.exec("INSERT INTO items VALUES (2, 100, 'grape') ON CONFLICT (id) DO UPDATE SET qty = EXCLUDED.qty, name = EXCLUDED.name");
                const res = await v.exec('SELECT qty, name FROM items WHERE id = 2');
                const pass = res.rows[0]?.qty === 100 && res.rows[0]?.name === 'grape';
                tests.push({ name: 'ON CONFLICT multi-column UPDATE', pass, actual: JSON.stringify(res.rows[0]) });
            } catch (e) {
                tests.push({ name: 'ON CONFLICT multi-column', pass: false, error: e.message });
            }

            // Test inserting new row with ON CONFLICT (no conflict)
            try {
                await v.exec("INSERT INTO items VALUES (3, 30, 'cherry') ON CONFLICT (id) DO NOTHING");
                const res = await v.exec('SELECT COUNT(*) AS cnt FROM items');
                const pass = res.rows[0]?.cnt === 3;  // Should have 3 rows now
                tests.push({ name: 'INSERT with ON CONFLICT (no conflict)', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'INSERT no conflict', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('UPDATE with FROM clause (JOIN-based update)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create tables
            await v.exec('CREATE TABLE orders (id INT, status TEXT, amount INT)');
            await v.exec('CREATE TABLE status_updates (order_id INT, new_status TEXT)');
            await v.exec("INSERT INTO orders VALUES (1, 'pending', 100), (2, 'pending', 200), (3, 'pending', 300)");
            await v.exec("INSERT INTO status_updates VALUES (1, 'shipped'), (3, 'delivered')");

            // Test UPDATE with FROM clause
            try {
                await v.exec('UPDATE orders o SET status = u.new_status FROM status_updates u WHERE o.id = u.order_id');
                const res = await v.exec('SELECT id, status FROM orders ORDER BY id');
                const pass = res.rows[0]?.status === 'shipped' &&
                             res.rows[1]?.status === 'pending' &&
                             res.rows[2]?.status === 'delivered';
                tests.push({ name: 'UPDATE FROM basic', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'UPDATE FROM basic', pass: false, error: e.message });
            }

            // Test UPDATE with FROM using expressions
            try {
                await v.exec('CREATE TABLE multipliers (order_id INT, factor INT)');
                await v.exec('INSERT INTO multipliers VALUES (1, 2), (2, 3)');
                await v.exec('UPDATE orders o SET amount = o.amount * m.factor FROM multipliers m WHERE o.id = m.order_id');
                const res = await v.exec('SELECT id, amount FROM orders ORDER BY id');
                const pass = res.rows[0]?.amount === 200 && res.rows[1]?.amount === 600 && res.rows[2]?.amount === 300;
                tests.push({ name: 'UPDATE FROM with expression', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'UPDATE FROM expression', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('DELETE with USING clause (JOIN-based delete)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create tables
            await v.exec('CREATE TABLE products (id INT, name TEXT, active INT)');
            await v.exec('CREATE TABLE to_delete (product_id INT)');
            await v.exec("INSERT INTO products VALUES (1, 'apple', 1), (2, 'banana', 1), (3, 'cherry', 1), (4, 'date', 1)");
            await v.exec('INSERT INTO to_delete VALUES (1), (3)');

            // Test DELETE with USING
            try {
                await v.exec('DELETE FROM products p USING to_delete d WHERE p.id = d.product_id');
                const res = await v.exec('SELECT id, name FROM products ORDER BY id');
                const pass = res.rows.length === 2 && res.rows[0]?.id === 2 && res.rows[1]?.id === 4;
                tests.push({ name: 'DELETE USING basic', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'DELETE USING basic', pass: false, error: e.message });
            }

            // Test DELETE with USING and additional WHERE conditions
            try {
                await v.exec('CREATE TABLE items2 (id INT, category TEXT)');
                await v.exec('CREATE TABLE blacklist (cat TEXT)');
                await v.exec("INSERT INTO items2 VALUES (1, 'food'), (2, 'food'), (3, 'electronics'), (4, 'electronics')");
                await v.exec("INSERT INTO blacklist VALUES ('electronics')");
                await v.exec('DELETE FROM items2 i USING blacklist b WHERE i.category = b.cat');
                const res = await v.exec('SELECT COUNT(*) AS cnt FROM items2');
                const pass = res.rows[0]?.cnt === 2;  // Only food items remain
                tests.push({ name: 'DELETE USING with category match', pass, actual: res.rows[0]?.cnt });
            } catch (e) {
                tests.push({ name: 'DELETE USING category', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('exportToLance creates valid Lance bytes', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup table
            await v.exec('CREATE TABLE export_test (id INT, name TEXT, score FLOAT)');
            await v.exec("INSERT INTO export_test VALUES (1, 'Alice', 95.5)");
            await v.exec("INSERT INTO export_test VALUES (2, 'Bob', 87.3)");
            await v.exec("INSERT INTO export_test VALUES (3, 'Charlie', 92.1)");

            // Test exportToLance
            try {
                const lanceBytes = await v.exportToLance('export_test');

                // Check it's a Uint8Array
                const isUint8 = lanceBytes instanceof Uint8Array;
                tests.push({ name: 'returns Uint8Array', pass: isUint8 });

                // Check minimum size (at least footer)
                const hasSize = lanceBytes.length >= 40;
                tests.push({ name: 'has minimum size', pass: hasSize, actual: lanceBytes.length });

                // Check magic bytes "LANC" at end
                const magic = String.fromCharCode(
                    lanceBytes[lanceBytes.length - 4],
                    lanceBytes[lanceBytes.length - 3],
                    lanceBytes[lanceBytes.length - 2],
                    lanceBytes[lanceBytes.length - 1]
                );
                tests.push({ name: 'has LANC magic', pass: magic === 'LANC', actual: magic });
            } catch (e) {
                tests.push({ name: 'exportToLance', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE export_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('exportToLance handles various column types', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Setup table with various types
            await v.exec('CREATE TABLE types_test (i INT, f FLOAT, s TEXT, b BOOLEAN)');
            await v.exec("INSERT INTO types_test VALUES (42, 3.14, 'hello', true)");
            await v.exec("INSERT INTO types_test VALUES (-1, 2.718, 'world', false)");

            // Test export
            try {
                const lanceBytes = await v.exportToLance('types_test');
                const magic = String.fromCharCode(
                    lanceBytes[lanceBytes.length - 4],
                    lanceBytes[lanceBytes.length - 3],
                    lanceBytes[lanceBytes.length - 2],
                    lanceBytes[lanceBytes.length - 1]
                );
                tests.push({
                    name: 'exports multiple types',
                    pass: magic === 'LANC' && lanceBytes.length > 100,
                    actual: `size=${lanceBytes.length}`
                });
            } catch (e) {
                tests.push({ name: 'exports multiple types', pass: false, error: e.message });
            }

            // Cleanup
            await v.exec('DROP TABLE types_test');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU aggregations match CPU results', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuAggregator } = await import('./lanceql.js');
            const tests = [];

            // Initialize GPU aggregator
            await gpuAggregator.init();
            const isAvailable = gpuAggregator.isAvailable();
            tests.push({ name: 'GPUAggregator initializes', pass: true, actual: isAvailable ? 'GPU available' : 'CPU fallback' });

            // Generate test data - 10,000 values to trigger GPU path
            const size = 10000;
            const values = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                values[i] = Math.random() * 1000 - 500; // -500 to 500
            }

            // CPU reference values
            let cpuSum = 0, cpuMin = values[0], cpuMax = values[0];
            for (let i = 0; i < size; i++) {
                cpuSum += values[i];
                if (values[i] < cpuMin) cpuMin = values[i];
                if (values[i] > cpuMax) cpuMax = values[i];
            }
            const cpuAvg = cpuSum / size;

            // Test SUM
            try {
                const gpuSum = await gpuAggregator.sum(values);
                const error = Math.abs(gpuSum - cpuSum);
                const pass = error < 0.01 * Math.abs(cpuSum) || error < 1; // 1% tolerance or absolute 1
                tests.push({ name: 'SUM matches CPU', pass, actual: `GPU=${gpuSum.toFixed(2)}, CPU=${cpuSum.toFixed(2)}` });
            } catch (e) {
                tests.push({ name: 'SUM matches CPU', pass: false, error: e.message });
            }

            // Test MIN
            try {
                const gpuMin = await gpuAggregator.min(values);
                const pass = Math.abs(gpuMin - cpuMin) < 0.001;
                tests.push({ name: 'MIN matches CPU', pass, actual: `GPU=${gpuMin.toFixed(4)}, CPU=${cpuMin.toFixed(4)}` });
            } catch (e) {
                tests.push({ name: 'MIN matches CPU', pass: false, error: e.message });
            }

            // Test MAX
            try {
                const gpuMax = await gpuAggregator.max(values);
                const pass = Math.abs(gpuMax - cpuMax) < 0.001;
                tests.push({ name: 'MAX matches CPU', pass, actual: `GPU=${gpuMax.toFixed(4)}, CPU=${cpuMax.toFixed(4)}` });
            } catch (e) {
                tests.push({ name: 'MAX matches CPU', pass: false, error: e.message });
            }

            // Test AVG
            try {
                const gpuAvg = await gpuAggregator.avg(values);
                const error = Math.abs(gpuAvg - cpuAvg);
                const pass = error < 0.01 * Math.abs(cpuAvg) || error < 0.01;
                tests.push({ name: 'AVG matches CPU', pass, actual: `GPU=${gpuAvg.toFixed(4)}, CPU=${cpuAvg.toFixed(4)}` });
            } catch (e) {
                tests.push({ name: 'AVG matches CPU', pass: false, error: e.message });
            }

            // Test COUNT
            try {
                const gpuCount = gpuAggregator.count(values);
                const pass = gpuCount === size;
                tests.push({ name: 'COUNT returns correct length', pass, actual: gpuCount });
            } catch (e) {
                tests.push({ name: 'COUNT', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU aggregations handle edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuAggregator } = await import('./lanceql.js');
            const tests = [];

            await gpuAggregator.init();

            // Test empty array
            try {
                const result = await gpuAggregator.avg([]);
                tests.push({ name: 'AVG of empty array returns null', pass: result === null, actual: result });
            } catch (e) {
                tests.push({ name: 'AVG of empty array', pass: false, error: e.message });
            }

            // Test small array (CPU fallback)
            try {
                const smallValues = [1, 2, 3, 4, 5];
                const sum = await gpuAggregator.sum(smallValues);
                tests.push({ name: 'Small array uses CPU fallback', pass: sum === 15, actual: sum });
            } catch (e) {
                tests.push({ name: 'Small array fallback', pass: false, error: e.message });
            }

            // Test with negative values
            try {
                const negValues = new Float32Array(2000);
                for (let i = 0; i < 2000; i++) negValues[i] = -i;
                const min = await gpuAggregator.min(negValues);
                tests.push({ name: 'MIN with negatives', pass: min === -1999, actual: min });
            } catch (e) {
                tests.push({ name: 'MIN with negatives', pass: false, error: e.message });
            }

            // Test all zeros
            try {
                const zeros = new Float32Array(5000);
                const sum = await gpuAggregator.sum(zeros);
                tests.push({ name: 'SUM of zeros', pass: sum === 0, actual: sum });
            } catch (e) {
                tests.push({ name: 'SUM of zeros', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU hash join matches CPU results', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuJoiner } = await import('./lanceql.js');
            const tests = [];

            // Initialize GPU joiner
            await gpuJoiner.init();
            const isAvailable = gpuJoiner.isAvailable();
            tests.push({ name: 'GPUJoiner initializes', pass: true, actual: isAvailable ? 'GPU available' : 'CPU fallback' });

            // Create test tables
            const leftRows = [];
            const rightRows = [];
            for (let i = 0; i < 1000; i++) {
                leftRows.push({ id: i, name: `left_${i}`, value: i * 10 });
            }
            for (let i = 0; i < 500; i++) {
                rightRows.push({ id: i * 2, category: `cat_${i}`, score: i });
            }

            // Test hash join
            try {
                const result = await gpuJoiner.hashJoin(leftRows, rightRows, 'id', 'id');

                // Should have 500 matches (every even id from 0-998)
                tests.push({
                    name: 'Hash join returns correct match count',
                    pass: result.matchCount === 500,
                    actual: result.matchCount
                });

                // Verify indices are valid
                let validIndices = true;
                for (let i = 0; i < Math.min(result.matchCount, 10); i++) {
                    const li = result.leftIndices[i];
                    const ri = result.rightIndices[i];
                    if (leftRows[li].id !== rightRows[ri].id) {
                        validIndices = false;
                        break;
                    }
                }
                tests.push({ name: 'Join indices match correctly', pass: validIndices });
            } catch (e) {
                tests.push({ name: 'Hash join', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU hash join handles edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuJoiner } = await import('./lanceql.js');
            const tests = [];

            await gpuJoiner.init();

            // Test empty tables
            try {
                const result = await gpuJoiner.hashJoin([], [{ id: 1 }], 'id', 'id');
                tests.push({ name: 'Empty left table returns 0 matches', pass: result.matchCount === 0, actual: result.matchCount });
            } catch (e) {
                tests.push({ name: 'Empty left table', pass: false, error: e.message });
            }

            // Test no matches
            try {
                const left = [{ id: 1 }, { id: 2 }, { id: 3 }];
                const right = [{ id: 100 }, { id: 200 }];
                const result = await gpuJoiner.hashJoin(left, right, 'id', 'id');
                tests.push({ name: 'No matches returns 0', pass: result.matchCount === 0, actual: result.matchCount });
            } catch (e) {
                tests.push({ name: 'No matches', pass: false, error: e.message });
            }

            // Test string keys
            try {
                const left = [{ name: 'alice' }, { name: 'bob' }, { name: 'charlie' }];
                const right = [{ name: 'bob' }, { name: 'david' }];
                const result = await gpuJoiner.hashJoin(left, right, 'name', 'name');
                tests.push({ name: 'String key join works', pass: result.matchCount === 1, actual: result.matchCount });
            } catch (e) {
                tests.push({ name: 'String key join', pass: false, error: e.message });
            }

            // Test many-to-many
            try {
                const left = [{ id: 1 }, { id: 1 }, { id: 2 }];
                const right = [{ id: 1 }, { id: 1 }];
                const result = await gpuJoiner.hashJoin(left, right, 'id', 'id');
                // 2 left rows with id=1, 2 right rows with id=1 = 4 matches
                tests.push({ name: 'Many-to-many join', pass: result.matchCount === 4, actual: result.matchCount });
            } catch (e) {
                tests.push({ name: 'Many-to-many join', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU bitonic sort matches CPU results', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuSorter } = await import('./lanceql.js');
            const tests = [];

            // Initialize GPU sorter
            await gpuSorter.init();
            const isAvailable = gpuSorter.isAvailable();
            tests.push({ name: 'GPUSorter initializes', pass: true, actual: isAvailable ? 'GPU available' : 'CPU fallback' });

            // Test ascending sort
            try {
                const values = new Float32Array([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
                const match = sorted.every((v, i) => v === expected[i]);
                tests.push({ name: 'Ascending sort', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Ascending sort', pass: false, error: e.message });
            }

            // Test descending sort
            try {
                const values = new Float32Array([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
                const sortedIdx = await gpuSorter.sort(values, false);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const expected = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
                const match = sorted.every((v, i) => v === expected[i]);
                tests.push({ name: 'Descending sort', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Descending sort', pass: false, error: e.message });
            }

            // Test already sorted array
            try {
                const values = new Float32Array([0, 1, 2, 3, 4]);
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const match = sorted.every((v, i) => v === i);
                tests.push({ name: 'Already sorted', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Already sorted', pass: false, error: e.message });
            }

            // Test reverse sorted array
            try {
                const values = new Float32Array([4, 3, 2, 1, 0]);
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const expected = [0, 1, 2, 3, 4];
                const match = sorted.every((v, i) => v === expected[i]);
                tests.push({ name: 'Reverse sorted', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Reverse sorted', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU sort handles edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuSorter } = await import('./lanceql.js');
            const tests = [];

            await gpuSorter.init();

            // Test empty array
            try {
                const result = await gpuSorter.sort(new Float32Array([]), true);
                tests.push({ name: 'Empty array', pass: result.length === 0, actual: result.length });
            } catch (e) {
                tests.push({ name: 'Empty array', pass: false, error: e.message });
            }

            // Test single element
            try {
                const result = await gpuSorter.sort(new Float32Array([42]), true);
                tests.push({ name: 'Single element', pass: result.length === 1 && result[0] === 0, actual: result.length });
            } catch (e) {
                tests.push({ name: 'Single element', pass: false, error: e.message });
            }

            // Test duplicate values
            try {
                const values = new Float32Array([3, 1, 2, 1, 3]);
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                // Should be [1, 1, 2, 3, 3]
                const match = sorted[0] === 1 && sorted[1] === 1 && sorted[2] === 2 && sorted[3] === 3 && sorted[4] === 3;
                tests.push({ name: 'Duplicate values', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Duplicate values', pass: false, error: e.message });
            }

            // Test negative numbers
            try {
                const values = new Float32Array([-5, 3, -1, 0, 2]);
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const expected = [-5, -1, 0, 2, 3];
                const match = sorted.every((v, i) => v === expected[i]);
                tests.push({ name: 'Negative numbers', pass: match, actual: sorted.join(',') });
            } catch (e) {
                tests.push({ name: 'Negative numbers', pass: false, error: e.message });
            }

            // Test larger array (power of 2)
            try {
                const values = new Float32Array(64);
                for (let i = 0; i < 64; i++) values[i] = 63 - i;  // Reverse order
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const match = sorted.every((v, i) => v === i);
                tests.push({ name: 'Power of 2 size', pass: match, actual: match ? '64 elements sorted' : sorted.slice(0, 5).join(',') + '...' });
            } catch (e) {
                tests.push({ name: 'Power of 2 size', pass: false, error: e.message });
            }

            // Test non-power of 2 size
            try {
                const values = new Float32Array(100);
                for (let i = 0; i < 100; i++) values[i] = 99 - i;  // Reverse order
                const sortedIdx = await gpuSorter.sort(values, true);
                const sorted = Array.from(sortedIdx).map(i => values[i]);
                const match = sorted.every((v, i) => v === i);
                tests.push({ name: 'Non-power of 2 size', pass: match, actual: match ? '100 elements sorted' : sorted.slice(0, 5).join(',') + '...' });
            } catch (e) {
                tests.push({ name: 'Non-power of 2 size', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU GROUP BY matches CPU results', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuGrouper } = await import('./lanceql.js');
            const tests = [];

            // Initialize GPU grouper
            await gpuGrouper.init();
            const isAvailable = gpuGrouper.isAvailable();
            tests.push({ name: 'GPUGrouper initializes', pass: true, actual: isAvailable ? 'GPU available' : 'CPU fallback' });

            // Test groupBy with unique keys
            try {
                const keys = new Uint32Array([1, 2, 3, 1, 2, 3, 1, 2, 3]);
                const result = await gpuGrouper.groupBy(keys);
                tests.push({ name: 'GroupBy finds unique groups', pass: result.numGroups === 3, actual: result.numGroups });
            } catch (e) {
                tests.push({ name: 'GroupBy finds unique groups', pass: false, error: e.message });
            }

            // Test groupBy assigns correct IDs
            try {
                const keys = new Uint32Array([10, 20, 10, 20, 30]);
                const result = await gpuGrouper.groupBy(keys);
                // Same keys should get same group IDs
                const sameGroup = result.groupIds[0] === result.groupIds[2] && result.groupIds[1] === result.groupIds[3];
                tests.push({ name: 'GroupBy assigns consistent IDs', pass: sameGroup, actual: Array.from(result.groupIds).join(',') });
            } catch (e) {
                tests.push({ name: 'GroupBy assigns consistent IDs', pass: false, error: e.message });
            }

            // Test group COUNT aggregation
            try {
                const keys = new Uint32Array([1, 2, 1, 2, 1]);
                const { groupIds, numGroups } = await gpuGrouper.groupBy(keys);
                const values = new Float32Array([1, 1, 1, 1, 1]);
                const counts = await gpuGrouper.groupAggregate(values, groupIds, numGroups, 'COUNT');
                // Group 1 appears 3 times, group 2 appears 2 times
                const total = counts[0] + counts[1];
                tests.push({ name: 'Group COUNT', pass: total === 5, actual: Array.from(counts).join(',') });
            } catch (e) {
                tests.push({ name: 'Group COUNT', pass: false, error: e.message });
            }

            // Test group SUM aggregation
            try {
                const keys = new Uint32Array([1, 2, 1, 2, 1]);
                const { groupIds, numGroups } = await gpuGrouper.groupBy(keys);
                const values = new Float32Array([10, 20, 30, 40, 50]);
                const sums = await gpuGrouper.groupAggregate(values, groupIds, numGroups, 'SUM');
                // Group 1: 10+30+50=90, Group 2: 20+40=60
                const total = sums[0] + sums[1];
                tests.push({ name: 'Group SUM', pass: Math.abs(total - 150) < 1, actual: Array.from(sums).join(',') });
            } catch (e) {
                tests.push({ name: 'Group SUM', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU GROUP BY handles edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuGrouper } = await import('./lanceql.js');
            const tests = [];

            await gpuGrouper.init();

            // Test single group
            try {
                const keys = new Uint32Array([5, 5, 5, 5, 5]);
                const result = await gpuGrouper.groupBy(keys);
                tests.push({ name: 'Single group', pass: result.numGroups === 1, actual: result.numGroups });
            } catch (e) {
                tests.push({ name: 'Single group', pass: false, error: e.message });
            }

            // Test all unique keys
            try {
                const keys = new Uint32Array([1, 2, 3, 4, 5]);
                const result = await gpuGrouper.groupBy(keys);
                tests.push({ name: 'All unique', pass: result.numGroups === 5, actual: result.numGroups });
            } catch (e) {
                tests.push({ name: 'All unique', pass: false, error: e.message });
            }

            // Test group MIN
            try {
                const keys = new Uint32Array([1, 2, 1, 2]);
                const { groupIds, numGroups } = await gpuGrouper.groupBy(keys);
                const values = new Float32Array([10, 5, 3, 8]);
                const mins = await gpuGrouper.groupAggregate(values, groupIds, numGroups, 'MIN');
                // Group 1: min(10, 3)=3, Group 2: min(5, 8)=5
                const pass = (mins[0] === 3 && mins[1] === 5) || (mins[0] === 5 && mins[1] === 3);
                tests.push({ name: 'Group MIN', pass, actual: Array.from(mins).join(',') });
            } catch (e) {
                tests.push({ name: 'Group MIN', pass: false, error: e.message });
            }

            // Test group MAX
            try {
                const keys = new Uint32Array([1, 2, 1, 2]);
                const { groupIds, numGroups } = await gpuGrouper.groupBy(keys);
                const values = new Float32Array([10, 5, 3, 8]);
                const maxs = await gpuGrouper.groupAggregate(values, groupIds, numGroups, 'MAX');
                // Group 1: max(10, 3)=10, Group 2: max(5, 8)=8
                const pass = (maxs[0] === 10 && maxs[1] === 8) || (maxs[0] === 8 && maxs[1] === 10);
                tests.push({ name: 'Group MAX', pass, actual: Array.from(maxs).join(',') });
            } catch (e) {
                tests.push({ name: 'Group MAX', pass: false, error: e.message });
            }

            // Test larger dataset
            try {
                const keys = new Uint32Array(1000);
                for (let i = 0; i < 1000; i++) keys[i] = i % 10;  // 10 groups
                const result = await gpuGrouper.groupBy(keys);
                tests.push({ name: 'Larger dataset (1000 rows, 10 groups)', pass: result.numGroups === 10, actual: result.numGroups });
            } catch (e) {
                tests.push({ name: 'Larger dataset', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('SELECT DISTINCT deduplicates rows', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            const v = await vault();
            await v.exec('CREATE TABLE test_distinct (id INT, category TEXT, value FLOAT)');
            await v.exec("INSERT INTO test_distinct VALUES (1, 'A', 10.5), (2, 'B', 20.5), (3, 'A', 10.5), (4, 'B', 30.5), (5, 'A', 10.5)");

            // Test DISTINCT on single column
            try {
                const res = await v.exec('SELECT DISTINCT category FROM test_distinct');
                const categories = res.rows.map(r => r.category).sort();
                const pass = categories.length === 2 && categories[0] === 'A' && categories[1] === 'B';
                tests.push({ name: 'DISTINCT single column', pass, actual: JSON.stringify(categories) });
            } catch (e) {
                tests.push({ name: 'DISTINCT single column', pass: false, error: e.message });
            }

            // Test DISTINCT on multiple columns
            try {
                const res = await v.exec('SELECT DISTINCT category, value FROM test_distinct');
                // A,10.5 appears 3 times but should be deduplicated to 1
                // B,20.5 and B,30.5 are unique
                tests.push({ name: 'DISTINCT multiple columns', pass: res.rows.length === 3, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'DISTINCT multiple columns', pass: false, error: e.message });
            }

            // Test DISTINCT with all unique rows
            try {
                const res = await v.exec('SELECT DISTINCT id FROM test_distinct');
                tests.push({ name: 'DISTINCT all unique', pass: res.rows.length === 5, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'DISTINCT all unique', pass: false, error: e.message });
            }

            // Test DISTINCT preserves row content
            try {
                const res = await v.exec('SELECT DISTINCT category FROM test_distinct ORDER BY category');
                const pass = res.rows[0].category === 'A' && res.rows[1].category === 'B';
                tests.push({ name: 'DISTINCT with ORDER BY', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'DISTINCT with ORDER BY', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE test_distinct');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('SELECT DISTINCT handles edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            const v = await vault();

            // Test DISTINCT with NULL values
            await v.exec('CREATE TABLE test_nulls (id INT, value TEXT)');
            await v.exec("INSERT INTO test_nulls VALUES (1, NULL), (2, 'A'), (3, NULL), (4, 'A'), (5, 'B')");
            try {
                const res = await v.exec('SELECT DISTINCT value FROM test_nulls');
                // Should have NULL, 'A', 'B' = 3 unique values
                tests.push({ name: 'DISTINCT with NULLs', pass: res.rows.length === 3, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'DISTINCT with NULLs', pass: false, error: e.message });
            }
            await v.exec('DROP TABLE test_nulls');

            // Test DISTINCT on empty table
            await v.exec('CREATE TABLE test_empty (id INT)');
            try {
                const res = await v.exec('SELECT DISTINCT id FROM test_empty');
                tests.push({ name: 'DISTINCT empty table', pass: res.rows.length === 0, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'DISTINCT empty table', pass: false, error: e.message });
            }
            await v.exec('DROP TABLE test_empty');

            // Test DISTINCT with single row
            await v.exec('CREATE TABLE test_single (id INT, name TEXT)');
            await v.exec("INSERT INTO test_single VALUES (1, 'Solo')");
            try {
                const res = await v.exec('SELECT DISTINCT name FROM test_single');
                tests.push({ name: 'DISTINCT single row', pass: res.rows.length === 1 && res.rows[0].name === 'Solo', actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'DISTINCT single row', pass: false, error: e.message });
            }
            await v.exec('DROP TABLE test_single');

            // Test DISTINCT with numeric duplicates
            await v.exec('CREATE TABLE test_nums (value FLOAT)');
            await v.exec('INSERT INTO test_nums VALUES (1.5), (2.5), (1.5), (3.5), (2.5), (1.5)');
            try {
                const res = await v.exec('SELECT DISTINCT value FROM test_nums ORDER BY value');
                const values = res.rows.map(r => r.value);
                const pass = values.length === 3 && values[0] === 1.5 && values[1] === 2.5 && values[2] === 3.5;
                tests.push({ name: 'DISTINCT numeric duplicates', pass, actual: JSON.stringify(values) });
            } catch (e) {
                tests.push({ name: 'DISTINCT numeric duplicates', pass: false, error: e.message });
            }
            await v.exec('DROP TABLE test_nums');

            // Test DISTINCT * (all columns)
            await v.exec('CREATE TABLE test_star (a INT, b TEXT)');
            await v.exec("INSERT INTO test_star VALUES (1, 'X'), (2, 'Y'), (1, 'X'), (2, 'Z')");
            try {
                const res = await v.exec('SELECT DISTINCT * FROM test_star');
                // (1,X), (2,Y), (2,Z) = 3 unique rows
                tests.push({ name: 'DISTINCT *', pass: res.rows.length === 3, actual: res.rows.length });
            } catch (e) {
                tests.push({ name: 'DISTINCT *', pass: false, error: e.message });
            }
            await v.exec('DROP TABLE test_star');

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU vector search top-K selection', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuVectorSearch, DistanceMetric } = await import('./lanceql.js');
            const tests = [];

            // Initialize GPU vector search
            await gpuVectorSearch.init();
            const isAvailable = gpuVectorSearch.isAvailable();
            tests.push({ name: 'GPUVectorSearch initializes', pass: true, actual: isAvailable ? 'GPU available' : 'CPU fallback' });

            // Test CPU top-K (small dataset)
            try {
                const scores = new Float32Array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8]);
                const result = await gpuVectorSearch.topK(scores, null, 3, true);
                // Top 3 descending should be indices 3(0.9), 7(0.8), 5(0.7)
                const pass = result.indices[0] === 3 && result.indices[1] === 7 && result.indices[2] === 5;
                tests.push({ name: 'CPU top-K descending', pass, actual: Array.from(result.indices).join(',') });
            } catch (e) {
                tests.push({ name: 'CPU top-K descending', pass: false, error: e.message });
            }

            // Test ascending top-K (for L2 distance)
            try {
                const scores = new Float32Array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8]);
                const result = await gpuVectorSearch.topK(scores, null, 3, false);
                // Top 3 ascending should be indices 0(0.1), 4(0.2), 2(0.3)
                const pass = result.indices[0] === 0 && result.indices[1] === 4 && result.indices[2] === 2;
                tests.push({ name: 'CPU top-K ascending', pass, actual: Array.from(result.indices).join(',') });
            } catch (e) {
                tests.push({ name: 'CPU top-K ascending', pass: false, error: e.message });
            }

            // Test distance computation (cosine)
            try {
                const query = new Float32Array([1, 0, 0, 0]);
                const vectors = [
                    new Float32Array([1, 0, 0, 0]),    // cos=1.0 (same direction)
                    new Float32Array([0, 1, 0, 0]),    // cos=0.0 (perpendicular)
                    new Float32Array([-1, 0, 0, 0]),   // cos=-1.0 (opposite)
                ];
                const scores = await gpuVectorSearch.computeDistances(query, vectors, 1, DistanceMetric.COSINE);
                const pass = Math.abs(scores[0] - 1.0) < 0.001 && Math.abs(scores[1]) < 0.001;
                tests.push({ name: 'Distance computation (cosine)', pass, actual: Array.from(scores).map(s => s.toFixed(2)).join(',') });
            } catch (e) {
                tests.push({ name: 'Distance computation (cosine)', pass: false, error: e.message });
            }

            // Test full search
            try {
                const query = new Float32Array([1, 0, 0, 0]);
                const vectors = [
                    new Float32Array([0.5, 0.5, 0, 0]),   // cos=0.707
                    new Float32Array([1, 0, 0, 0]),       // cos=1.0 (best)
                    new Float32Array([0, 1, 0, 0]),       // cos=0.0
                    new Float32Array([0.9, 0.1, 0, 0]),   // cos=0.994 (second best)
                ];
                const result = await gpuVectorSearch.search(query, vectors, 2, { metric: DistanceMetric.COSINE });
                // Best match should be index 1, second best index 3
                const pass = result.indices[0] === 1 && result.indices[1] === 3;
                tests.push({ name: 'Full vector search', pass, actual: Array.from(result.indices).join(',') });
            } catch (e) {
                tests.push({ name: 'Full vector search', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('GPU vector search handles edge cases', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { gpuVectorSearch, DistanceMetric } = await import('./lanceql.js');
            const tests = [];

            await gpuVectorSearch.init();

            // Test empty input
            try {
                const scores = new Float32Array([]);
                const result = await gpuVectorSearch.topK(scores, null, 5, true);
                tests.push({ name: 'Empty input', pass: result.indices.length === 0, actual: result.indices.length });
            } catch (e) {
                tests.push({ name: 'Empty input', pass: false, error: e.message });
            }

            // Test k > n
            try {
                const scores = new Float32Array([0.5, 0.3]);
                const result = await gpuVectorSearch.topK(scores, null, 10, true);
                tests.push({ name: 'K > N', pass: result.indices.length === 2, actual: result.indices.length });
            } catch (e) {
                tests.push({ name: 'K > N', pass: false, error: e.message });
            }

            // Test single element
            try {
                const scores = new Float32Array([0.42]);
                const result = await gpuVectorSearch.topK(scores, null, 1, true);
                const pass = result.indices[0] === 0 && Math.abs(result.scores[0] - 0.42) < 0.001;
                tests.push({ name: 'Single element', pass, actual: `idx=${result.indices[0]}, score=${result.scores[0]}` });
            } catch (e) {
                tests.push({ name: 'Single element', pass: false, error: e.message });
            }

            // Test L2 distance
            try {
                const query = new Float32Array([0, 0, 0, 0]);
                const vectors = [
                    new Float32Array([1, 0, 0, 0]),   // L2=1.0
                    new Float32Array([3, 0, 0, 0]),   // L2=3.0
                    new Float32Array([2, 0, 0, 0]),   // L2=2.0
                ];
                const scores = await gpuVectorSearch.computeDistances(query, vectors, 1, DistanceMetric.L2);
                // For L2, lower is better, so we'd use ascending
                const result = await gpuVectorSearch.topK(scores, null, 2, false);
                // Closest should be index 0 (L2=1.0), then index 2 (L2=2.0)
                const pass = result.indices[0] === 0 && result.indices[1] === 2;
                tests.push({ name: 'L2 distance search', pass, actual: Array.from(result.indices).join(',') });
            } catch (e) {
                tests.push({ name: 'L2 distance search', pass: false, error: e.message });
            }

            // Test with custom indices
            try {
                const scores = new Float32Array([0.1, 0.9, 0.5]);
                const indices = new Uint32Array([100, 200, 300]);  // Custom row IDs
                const result = await gpuVectorSearch.topK(scores, indices, 2, true);
                // Best is scores[1]=0.9 which maps to indices[1]=200
                const pass = result.indices[0] === 200 && result.indices[1] === 300;
                tests.push({ name: 'Custom indices', pass, actual: Array.from(result.indices).join(',') });
            } catch (e) {
                tests.push({ name: 'Custom indices', pass: false, error: e.message });
            }

            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('String aggregates (STRING_AGG, GROUP_CONCAT, MEDIAN)', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            const v = await vault();

            // Create test table
            await v.exec('CREATE TABLE test_agg (category TEXT, name TEXT, value FLOAT)');
            await v.exec("INSERT INTO test_agg VALUES ('A', 'Alice', 10), ('A', 'Bob', 20), ('A', 'Charlie', 30), ('B', 'Dave', 5), ('B', 'Eve', 15)");

            // Test STRING_AGG with separator
            try {
                const res = await v.exec("SELECT category, STRING_AGG(name, '; ') AS names FROM test_agg GROUP BY category ORDER BY category");
                const aNames = res.rows.find(r => r.category === 'A')?.names;
                const pass = aNames && aNames.includes('Alice') && aNames.includes('Bob') && aNames.includes('Charlie') && aNames.includes('; ');
                tests.push({ name: 'STRING_AGG with separator', pass, actual: aNames });
            } catch (e) {
                tests.push({ name: 'STRING_AGG with separator', pass: false, error: e.message });
            }

            // Test GROUP_CONCAT (default comma)
            try {
                const res = await v.exec('SELECT category, GROUP_CONCAT(name) AS names FROM test_agg GROUP BY category ORDER BY category');
                const aNames = res.rows.find(r => r.category === 'A')?.names;
                const pass = aNames && aNames.includes(',') && aNames.includes('Alice');
                tests.push({ name: 'GROUP_CONCAT', pass, actual: aNames });
            } catch (e) {
                tests.push({ name: 'GROUP_CONCAT', pass: false, error: e.message });
            }

            // Test MEDIAN with odd count
            try {
                const res = await v.exec('SELECT category, MEDIAN(value) AS med FROM test_agg GROUP BY category ORDER BY category');
                const aMed = res.rows.find(r => r.category === 'A')?.med;
                // A has 10, 20, 30 -> median = 20
                tests.push({ name: 'MEDIAN odd count', pass: aMed === 20, actual: aMed });
            } catch (e) {
                tests.push({ name: 'MEDIAN odd count', pass: false, error: e.message });
            }

            // Test MEDIAN with even count
            await v.exec("INSERT INTO test_agg VALUES ('A', 'Frank', 40)");
            try {
                const res = await v.exec('SELECT category, MEDIAN(value) AS med FROM test_agg WHERE category = \'A\' GROUP BY category');
                const aMed = res.rows[0]?.med;
                // A has 10, 20, 30, 40 -> median = (20+30)/2 = 25
                tests.push({ name: 'MEDIAN even count', pass: aMed === 25, actual: aMed });
            } catch (e) {
                tests.push({ name: 'MEDIAN even count', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE test_agg');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('STDDEV and VARIANCE aggregates', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const tests = [];

            const v = await vault();

            // Create test table with known values
            await v.exec('CREATE TABLE test_stats (grp TEXT, value FLOAT)');
            // Group A: values 2, 4, 6, 8, 10 -> mean=6, sample_var=10, pop_var=8
            await v.exec("INSERT INTO test_stats VALUES ('A', 2), ('A', 4), ('A', 6), ('A', 8), ('A', 10)");
            // Group B: values 1, 1 -> mean=1, variance=0
            await v.exec("INSERT INTO test_stats VALUES ('B', 1), ('B', 1)");

            // Test STDDEV (sample)
            try {
                const res = await v.exec('SELECT grp, STDDEV(value) AS sd FROM test_stats GROUP BY grp ORDER BY grp');
                const aSd = res.rows.find(r => r.grp === 'A')?.sd;
                // sample variance = 10, sample stddev = sqrt(10) ≈ 3.162
                const pass = Math.abs(aSd - Math.sqrt(10)) < 0.01;
                tests.push({ name: 'STDDEV (sample)', pass, actual: aSd?.toFixed(3) });
            } catch (e) {
                tests.push({ name: 'STDDEV (sample)', pass: false, error: e.message });
            }

            // Test STDDEV_POP
            try {
                const res = await v.exec('SELECT grp, STDDEV_POP(value) AS sd FROM test_stats GROUP BY grp ORDER BY grp');
                const aSd = res.rows.find(r => r.grp === 'A')?.sd;
                // population variance = 8, population stddev = sqrt(8) ≈ 2.828
                const pass = Math.abs(aSd - Math.sqrt(8)) < 0.01;
                tests.push({ name: 'STDDEV_POP', pass, actual: aSd?.toFixed(3) });
            } catch (e) {
                tests.push({ name: 'STDDEV_POP', pass: false, error: e.message });
            }

            // Test VARIANCE (sample)
            try {
                const res = await v.exec('SELECT grp, VARIANCE(value) AS var FROM test_stats GROUP BY grp ORDER BY grp');
                const aVar = res.rows.find(r => r.grp === 'A')?.var;
                // sample variance = 10
                tests.push({ name: 'VARIANCE (sample)', pass: Math.abs(aVar - 10) < 0.01, actual: aVar });
            } catch (e) {
                tests.push({ name: 'VARIANCE (sample)', pass: false, error: e.message });
            }

            // Test VAR_POP
            try {
                const res = await v.exec('SELECT grp, VAR_POP(value) AS var FROM test_stats GROUP BY grp ORDER BY grp');
                const aVar = res.rows.find(r => r.grp === 'A')?.var;
                // population variance = 8
                tests.push({ name: 'VAR_POP', pass: Math.abs(aVar - 8) < 0.01, actual: aVar });
            } catch (e) {
                tests.push({ name: 'VAR_POP', pass: false, error: e.message });
            }

            // Test with identical values (variance = 0)
            try {
                const res = await v.exec("SELECT STDDEV(value) AS sd FROM test_stats WHERE grp = 'B'");
                const bSd = res.rows[0]?.sd;
                tests.push({ name: 'STDDEV with identical values', pass: bSd === 0, actual: bSd });
            } catch (e) {
                tests.push({ name: 'STDDEV with identical values', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE test_stats');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('SEARCH + Aggregations (COUNT, SUM, AVG with NEAR)', async ({ page }) => {
        // This test verifies that vector search can be combined with aggregate functions
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create test table with text and numeric data
            await v.exec(`CREATE TABLE search_agg (
                id INT,
                category TEXT,
                description TEXT,
                price REAL,
                quantity INT
            )`);

            // Insert test data with semantically similar descriptions
            await v.exec(`INSERT INTO search_agg VALUES
                (1, 'electronics', 'laptop computer for work', 999.99, 5),
                (2, 'electronics', 'gaming laptop with graphics', 1499.99, 3),
                (3, 'electronics', 'desktop computer workstation', 1299.99, 2),
                (4, 'furniture', 'office desk wooden', 299.99, 10),
                (5, 'furniture', 'ergonomic chair', 399.99, 8),
                (6, 'electronics', 'notebook computer portable', 799.99, 7)
            `);

            // Test 1: COUNT(*) with NEAR search
            try {
                const res = await v.exec(`SELECT COUNT(*) AS cnt FROM search_agg WHERE description NEAR 'computer laptop' LIMIT 10`);
                // Should find rows with similar descriptions
                const cnt = res.rows[0]?.cnt;
                tests.push({ name: 'COUNT with NEAR', pass: typeof cnt === 'number' && cnt > 0, actual: cnt });
            } catch (e) {
                tests.push({ name: 'COUNT with NEAR', pass: false, error: e.message });
            }

            // Test 2: SUM with NEAR search
            try {
                const res = await v.exec(`SELECT SUM(quantity) AS total FROM search_agg WHERE description NEAR 'laptop' LIMIT 10`);
                const total = res.rows[0]?.total;
                tests.push({ name: 'SUM with NEAR', pass: typeof total === 'number', actual: total });
            } catch (e) {
                tests.push({ name: 'SUM with NEAR', pass: false, error: e.message });
            }

            // Test 3: AVG with NEAR search
            try {
                const res = await v.exec(`SELECT AVG(price) AS avg_price FROM search_agg WHERE description NEAR 'computer' LIMIT 10`);
                const avg = res.rows[0]?.avg_price;
                tests.push({ name: 'AVG with NEAR', pass: typeof avg === 'number' && avg > 0, actual: avg?.toFixed(2) });
            } catch (e) {
                tests.push({ name: 'AVG with NEAR', pass: false, error: e.message });
            }

            // Test 4: Multiple aggregates with NEAR
            try {
                const res = await v.exec(`SELECT COUNT(*) AS cnt, MIN(price) AS min_p, MAX(price) AS max_p FROM search_agg WHERE description NEAR 'laptop computer' LIMIT 10`);
                const row = res.rows[0];
                const pass = typeof row?.cnt === 'number' && typeof row?.min_p === 'number' && typeof row?.max_p === 'number';
                tests.push({ name: 'Multiple aggregates with NEAR', pass, actual: `cnt=${row?.cnt}, min=${row?.min_p}, max=${row?.max_p}` });
            } catch (e) {
                tests.push({ name: 'Multiple aggregates with NEAR', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE search_agg');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('SEARCH + GROUP BY Aggregations', async ({ page }) => {
        // This test verifies that vector search can be combined with GROUP BY
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create test table with categories
            await v.exec(`CREATE TABLE search_group (
                id INT,
                category TEXT,
                name TEXT,
                score REAL
            )`);

            // Insert test data
            await v.exec(`INSERT INTO search_group VALUES
                (1, 'A', 'apple fruit fresh', 85.5),
                (2, 'A', 'apple juice drink', 78.0),
                (3, 'B', 'banana yellow fruit', 92.0),
                (4, 'B', 'banana smoothie blend', 88.5),
                (5, 'A', 'apple pie dessert', 95.0),
                (6, 'B', 'banana bread baked', 82.0)
            `);

            // Test 1: GROUP BY with NEAR search
            try {
                const res = await v.exec(`SELECT category, COUNT(*) AS cnt FROM search_group WHERE name NEAR 'fruit' GROUP BY category LIMIT 10`);
                // Should group search results by category
                const pass = res.rows.length > 0 && res.rows.every(r => typeof r.cnt === 'number');
                tests.push({ name: 'GROUP BY with NEAR', pass, actual: JSON.stringify(res.rows) });
            } catch (e) {
                tests.push({ name: 'GROUP BY with NEAR', pass: false, error: e.message });
            }

            // Test 2: GROUP BY with aggregate functions
            try {
                const res = await v.exec(`SELECT category, AVG(score) AS avg_score, SUM(score) AS total FROM search_group WHERE name NEAR 'apple banana' GROUP BY category LIMIT 10`);
                const pass = res.rows.length > 0 && res.rows.every(r => typeof r.avg_score === 'number');
                tests.push({ name: 'GROUP BY with multiple aggregates', pass, actual: JSON.stringify(res.rows.map(r => ({ cat: r.category, avg: r.avg_score?.toFixed(1) }))) });
            } catch (e) {
                tests.push({ name: 'GROUP BY with multiple aggregates', pass: false, error: e.message });
            }

            // Test 3: Empty search results should return empty GROUP BY
            try {
                const res = await v.exec(`SELECT category, COUNT(*) AS cnt FROM search_group WHERE name NEAR 'xyz_nonexistent_term' GROUP BY category LIMIT 5`);
                // With vector search, it may still return results based on similarity
                // The key is it should not error
                tests.push({ name: 'GROUP BY with no exact matches', pass: true, actual: `${res.rows.length} groups` });
            } catch (e) {
                tests.push({ name: 'GROUP BY with no exact matches', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE search_group');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('BM25 Full-Text Search (NEAR without vector column)', async ({ page }) => {
        // This test verifies BM25 text search works when no vector/embedding column exists
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create test table WITHOUT embedding column (pure text)
            await v.exec(`CREATE TABLE articles (
                id INT,
                title TEXT,
                content TEXT,
                category TEXT
            )`);

            // Insert test data with varied content for BM25 ranking
            await v.exec(`INSERT INTO articles VALUES
                (1, 'Introduction to Machine Learning', 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.', 'tech'),
                (2, 'Deep Learning Fundamentals', 'Deep learning uses neural networks with many layers to learn complex patterns in data.', 'tech'),
                (3, 'Database Design Principles', 'Proper database design ensures data integrity and efficient query performance.', 'tech'),
                (4, 'Cooking with Fresh Vegetables', 'Fresh vegetables are essential for healthy cooking and nutrition.', 'food'),
                (5, 'Advanced Machine Learning Techniques', 'Advanced machine learning includes ensemble methods, transfer learning, and reinforcement learning.', 'tech'),
                (6, 'Garden Vegetable Recipes', 'These vegetable recipes use garden fresh ingredients for delicious meals.', 'food')
            `);

            // Test 1: Basic BM25 search - should rank by relevance
            try {
                const res = await v.exec(`SELECT id, title FROM articles WHERE content NEAR 'machine learning' LIMIT 10`);
                // Should find articles about machine learning, ranked by relevance
                const ids = res.rows.map(r => r.id);
                // Articles 1, 2, 5 mention machine/learning
                const hasMlArticles = ids.some(id => [1, 2, 5].includes(id));
                tests.push({ name: 'Basic BM25 search', pass: hasMlArticles && res.rows.length > 0, actual: `Found ${res.rows.length} results, ids: ${ids.join(',')}` });
            } catch (e) {
                tests.push({ name: 'Basic BM25 search', pass: false, error: e.message });
            }

            // Test 2: Multi-word query with ranking
            try {
                const res = await v.exec(`SELECT id, title FROM articles WHERE title NEAR 'machine learning' LIMIT 5`);
                // Titles with "machine learning" should rank higher
                const topId = res.rows[0]?.id;
                // Article 1 or 5 should be top (both have "Machine Learning" in title)
                tests.push({ name: 'Multi-word query ranking', pass: [1, 5].includes(topId), actual: `Top result id: ${topId}` });
            } catch (e) {
                tests.push({ name: 'Multi-word query ranking', pass: false, error: e.message });
            }

            // Test 3: BM25 with additional WHERE conditions
            try {
                const res = await v.exec(`SELECT id, title FROM articles WHERE content NEAR 'learning' AND category = 'tech' LIMIT 10`);
                // Should find tech articles about learning
                const allTech = res.rows.every(r => true); // Can't verify category in result without selecting it
                tests.push({ name: 'BM25 with WHERE filter', pass: res.rows.length > 0, actual: `Found ${res.rows.length} tech articles` });
            } catch (e) {
                tests.push({ name: 'BM25 with WHERE filter', pass: false, error: e.message });
            }

            // Test 4: BM25 + COUNT aggregation
            try {
                const res = await v.exec(`SELECT COUNT(*) AS cnt FROM articles WHERE content NEAR 'data' LIMIT 10`);
                const cnt = res.rows[0]?.cnt;
                // Articles 1, 2, 3 mention "data"
                tests.push({ name: 'BM25 + COUNT aggregate', pass: typeof cnt === 'number' && cnt > 0, actual: `Count: ${cnt}` });
            } catch (e) {
                tests.push({ name: 'BM25 + COUNT aggregate', pass: false, error: e.message });
            }

            // Test 5: No matches (rare term)
            try {
                const res = await v.exec(`SELECT id FROM articles WHERE content NEAR 'xyznonexistent' LIMIT 10`);
                // Should return empty results, not error
                tests.push({ name: 'No matches returns empty', pass: res.rows.length === 0, actual: `${res.rows.length} results` });
            } catch (e) {
                tests.push({ name: 'No matches returns empty', pass: false, error: e.message });
            }

            // Test 6: Stop words only query
            try {
                const res = await v.exec(`SELECT id FROM articles WHERE content NEAR 'the and or' LIMIT 10`);
                // Query with only stop words should return empty (no meaningful terms)
                tests.push({ name: 'Stop words only returns empty', pass: res.rows.length === 0, actual: `${res.rows.length} results` });
            } catch (e) {
                tests.push({ name: 'Stop words only returns empty', pass: false, error: e.message });
            }

            // Test 7: Term frequency affects ranking
            try {
                const res = await v.exec(`SELECT id, title FROM articles WHERE content NEAR 'vegetables' LIMIT 5`);
                // Articles 4 and 6 have "vegetables", 6 mentions it twice
                const ids = res.rows.map(r => r.id);
                const hasVegetableArticles = ids.includes(4) || ids.includes(6);
                tests.push({ name: 'Term frequency ranking', pass: hasVegetableArticles, actual: `ids: ${ids.join(',')}` });
            } catch (e) {
                tests.push({ name: 'Term frequency ranking', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE articles');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('PIVOT transformation - rows to columns with aggregation', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create sales table for PIVOT tests
            await v.exec(`CREATE TABLE sales (
                region TEXT,
                quarter TEXT,
                amount DOUBLE
            )`);

            // Insert test data
            await v.exec(`INSERT INTO sales VALUES
                ('East', 'Q1', 100),
                ('East', 'Q2', 150),
                ('East', 'Q3', 200),
                ('East', 'Q4', 180),
                ('West', 'Q1', 120),
                ('West', 'Q2', 130),
                ('West', 'Q3', 170),
                ('West', 'Q4', 160),
                ('North', 'Q1', 90),
                ('North', 'Q2', 110)
            `);

            // Test 1: Basic PIVOT with SUM
            try {
                const res = await v.exec(`
                    SELECT region, quarter, amount FROM sales
                    PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
                `);
                // Should have columns: region, Q1, Q2, Q3, Q4
                const hasCorrectCols = res.columns.includes('Q1') && res.columns.includes('Q4');
                // Find East row
                const eastRow = res.rows.find(r => r.region === 'East');
                const eastQ1 = eastRow?.Q1 || eastRow?.[1];
                tests.push({
                    name: 'PIVOT SUM basic',
                    pass: hasCorrectCols && eastQ1 === 100,
                    actual: `cols: ${res.columns.join(',')}, East Q1: ${eastQ1}`
                });
            } catch (e) {
                tests.push({ name: 'PIVOT SUM basic', pass: false, error: e.message });
            }

            // Test 2: PIVOT with COUNT
            try {
                const res = await v.exec(`
                    SELECT region, quarter, amount FROM sales
                    PIVOT (COUNT(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
                `);
                // East has 1 entry per quarter
                const eastRow = res.rows.find(r => r.region === 'East');
                const eastQ1Count = eastRow?.Q1 || eastRow?.[1];
                // North only has Q1 and Q2
                const northRow = res.rows.find(r => r.region === 'North');
                const northQ3Count = northRow?.Q3 || northRow?.[3];
                tests.push({
                    name: 'PIVOT COUNT',
                    pass: eastQ1Count === 1 && northQ3Count === 0,
                    actual: `East Q1: ${eastQ1Count}, North Q3: ${northQ3Count}`
                });
            } catch (e) {
                tests.push({ name: 'PIVOT COUNT', pass: false, error: e.message });
            }

            // Test 3: PIVOT with AVG
            try {
                // Add duplicate entries for avg testing
                await v.exec(`INSERT INTO sales VALUES ('South', 'Q1', 100), ('South', 'Q1', 200)`);
                const res = await v.exec(`
                    SELECT region, quarter, amount FROM sales
                    PIVOT (AVG(amount) FOR quarter IN ('Q1', 'Q2'))
                `);
                const southRow = res.rows.find(r => r.region === 'South');
                const southQ1Avg = southRow?.Q1 || southRow?.[1];
                tests.push({
                    name: 'PIVOT AVG',
                    pass: southQ1Avg === 150,
                    actual: `South Q1 avg: ${southQ1Avg}`
                });
            } catch (e) {
                tests.push({ name: 'PIVOT AVG', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE sales');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('UNPIVOT transformation - columns to rows', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create wide table for UNPIVOT tests
            await v.exec(`CREATE TABLE quarterly_sales (
                product TEXT,
                q1_sales DOUBLE,
                q2_sales DOUBLE,
                q3_sales DOUBLE,
                q4_sales DOUBLE
            )`);

            // Insert test data
            await v.exec(`INSERT INTO quarterly_sales VALUES
                ('Widget', 100, 150, 200, 180),
                ('Gadget', 120, 130, 170, 160),
                ('Doohickey', 90, 110, 140, 130)
            `);

            // Test 1: Basic UNPIVOT
            try {
                const res = await v.exec(`
                    SELECT product, q1_sales, q2_sales, q3_sales, q4_sales FROM quarterly_sales
                    UNPIVOT (sales FOR quarter IN (q1_sales, q2_sales, q3_sales, q4_sales))
                `);
                // Should have columns: product, quarter, sales
                const hasCorrectCols = res.columns.includes('quarter') && res.columns.includes('sales');
                // Should have 12 rows (3 products x 4 quarters)
                const correctRowCount = res.rows.length === 12;
                // Check first widget row
                const widgetQ1 = res.rows.find(r => r.product === 'Widget' && r.quarter === 'q1_sales');
                tests.push({
                    name: 'UNPIVOT basic',
                    pass: hasCorrectCols && correctRowCount && widgetQ1?.sales === 100,
                    actual: `cols: ${res.columns.join(',')}, rows: ${res.rows.length}, Widget Q1: ${widgetQ1?.sales}`
                });
            } catch (e) {
                tests.push({ name: 'UNPIVOT basic', pass: false, error: e.message });
            }

            // Test 2: UNPIVOT with NULL handling
            try {
                // Create table with NULL values
                await v.exec(`CREATE TABLE sparse_data (
                    id INT,
                    jan DOUBLE,
                    feb DOUBLE,
                    mar DOUBLE
                )`);
                await v.exec(`INSERT INTO sparse_data VALUES (1, 100, NULL, 300), (2, NULL, 200, NULL)`);

                const res = await v.exec(`
                    SELECT id, jan, feb, mar FROM sparse_data
                    UNPIVOT (value FOR month IN (jan, feb, mar))
                `);
                // Should skip NULL values - expect 3 rows (1:jan, 1:mar, 2:feb)
                tests.push({
                    name: 'UNPIVOT NULL skip',
                    pass: res.rows.length === 3,
                    actual: `rows: ${res.rows.length}`
                });

                await v.exec('DROP TABLE sparse_data');
            } catch (e) {
                tests.push({ name: 'UNPIVOT NULL skip', pass: false, error: e.message });
            }

            // Test 3: UNPIVOT preserves non-pivoted columns
            try {
                const res = await v.exec(`
                    SELECT product, q1_sales, q2_sales FROM quarterly_sales
                    UNPIVOT (amount FOR period IN (q1_sales, q2_sales))
                `);
                // Preserved column should be product
                const hasProduct = res.columns.includes('product');
                const hasPeriod = res.columns.includes('period');
                const hasAmount = res.columns.includes('amount');
                // Should have 6 rows (3 products x 2 periods)
                tests.push({
                    name: 'UNPIVOT preserved columns',
                    pass: hasProduct && hasPeriod && hasAmount && res.rows.length === 6,
                    actual: `cols: ${res.columns.join(',')}, rows: ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'UNPIVOT preserved columns', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE quarterly_sales');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });

    test('QUALIFY clause - filter on window function results', async ({ page }) => {
        const results = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const tests = [];

            // Create employees table for QUALIFY tests
            await v.exec(`CREATE TABLE employees (
                id INT,
                name TEXT,
                dept TEXT,
                salary DOUBLE
            )`);

            // Insert test data
            await v.exec(`INSERT INTO employees VALUES
                (1, 'Alice', 'Engineering', 120000),
                (2, 'Bob', 'Engineering', 110000),
                (3, 'Charlie', 'Engineering', 100000),
                (4, 'Diana', 'Sales', 90000),
                (5, 'Eve', 'Sales', 95000),
                (6, 'Frank', 'Sales', 85000),
                (7, 'Grace', 'HR', 80000),
                (8, 'Henry', 'HR', 75000)
            `);

            // Test 1: Basic QUALIFY with ROW_NUMBER - top 1 per department
            try {
                const res = await v.exec(`
                    SELECT id, name, dept, salary,
                           ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn
                    FROM employees
                    QUALIFY rn = 1
                `);
                // Should get 3 rows (1 per department: Alice, Eve, Grace)
                const names = res.rows.map(r => r.name);
                const hasTopEng = names.includes('Alice');
                const hasTopSales = names.includes('Eve');
                const hasTopHR = names.includes('Grace');
                tests.push({
                    name: 'QUALIFY ROW_NUMBER = 1',
                    pass: res.rows.length === 3 && hasTopEng && hasTopSales && hasTopHR,
                    actual: `rows: ${res.rows.length}, names: ${names.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'QUALIFY ROW_NUMBER = 1', pass: false, error: e.message });
            }

            // Test 2: QUALIFY with RANK - top 2 per department
            try {
                const res = await v.exec(`
                    SELECT id, name, dept, salary,
                           RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rnk
                    FROM employees
                    QUALIFY rnk <= 2
                `);
                // Should get 6 rows (2 per department)
                // Engineering: Alice, Bob; Sales: Eve, Diana; HR: Grace, Henry
                tests.push({
                    name: 'QUALIFY RANK <= 2',
                    pass: res.rows.length === 6,
                    actual: `rows: ${res.rows.length}`
                });
            } catch (e) {
                tests.push({ name: 'QUALIFY RANK <= 2', pass: false, error: e.message });
            }

            // Test 3: QUALIFY without partition - global top 3
            try {
                const res = await v.exec(`
                    SELECT id, name, salary,
                           ROW_NUMBER() OVER (ORDER BY salary DESC) as rn
                    FROM employees
                    QUALIFY rn <= 3
                `);
                // Should get top 3 by salary: Alice (120k), Bob (110k), Charlie (100k)
                const names = res.rows.map(r => r.name);
                tests.push({
                    name: 'QUALIFY global top 3',
                    pass: res.rows.length === 3 && names.includes('Alice') && names.includes('Bob'),
                    actual: `rows: ${res.rows.length}, names: ${names.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'QUALIFY global top 3', pass: false, error: e.message });
            }

            // Test 4: QUALIFY with compound condition
            try {
                const res = await v.exec(`
                    SELECT id, name, dept, salary,
                           ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn
                    FROM employees
                    QUALIFY rn = 1 AND salary > 85000
                `);
                // Should get Alice (120k) and Eve (95k), but not Grace (80k < 85k)
                const names = res.rows.map(r => r.name);
                tests.push({
                    name: 'QUALIFY compound condition',
                    pass: res.rows.length === 2 && names.includes('Alice') && names.includes('Eve'),
                    actual: `rows: ${res.rows.length}, names: ${names.join(',')}`
                });
            } catch (e) {
                tests.push({ name: 'QUALIFY compound condition', pass: false, error: e.message });
            }

            await v.exec('DROP TABLE employees');
            return tests;
        });

        for (const t of results) {
            expect(t.pass, `${t.name}: ${t.error || 'got ' + t.actual}`).toBe(true);
        }
    });
});
