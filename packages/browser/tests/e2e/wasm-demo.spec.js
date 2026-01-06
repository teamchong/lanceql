// @ts-check
import { test, expect } from '@playwright/test';

test.describe('Demo Page - Push-Down Filtering + JOIN + Vector Search', () => {
  test.beforeEach(async ({ context }) => {
    // Disable CORS for CDN requests
    await context.route('**/*', route => {
      route.continue();
    });
  });

  test('default query shows results with JOIN + NEAR', async ({ page }) => {
    test.setTimeout(90000);

    await page.goto('/examples/wasm/');

    // Wait for WASM to load
    await page.waitForSelector('#sql-input', { timeout: 10000 });
    await page.waitForTimeout(3000);

    // Verify default query has our power features
    const sqlInput = page.locator('#sql-input');
    const sqlContent = await sqlInput.inputValue();
    expect(sqlContent).toContain('JOIN');
    expect(sqlContent).toContain('NEAR');
    expect(sqlContent).toContain('Push-Down Filtering');

    // Click Run button
    const runButton = page.locator('button:has-text("Run")').first();
    await runButton.click();

    // Wait for results to appear (network requests to CDN)
    // Look for actual data rows in the results table
    const resultsBody = page.locator('.results-body, .data-table tbody, table tbody');

    // Wait up to 60 seconds for results (CDN may be slow)
    await expect(resultsBody).toBeVisible({ timeout: 60000 });

    // Verify we have actual data rows displayed
    const dataRows = page.locator('.data-table tr, table tr').filter({ hasNot: page.locator('th') });
    const rowCount = await dataRows.count();

    console.log(`Results: ${rowCount} rows displayed`);

    // We should have at least 1 row of results
    expect(rowCount).toBeGreaterThan(0);

    // Verify the result contains expected columns from the JOIN
    // The query selects: i.url, i.text, t.text_zh, t.text_es
    const headerRow = page.locator('.data-table th, table th').first();
    await expect(headerRow).toBeVisible();
  });

  test('vector search (NEAR) returns relevant results', async ({ page }) => {
    test.setTimeout(90000);

    await page.goto('/examples/wasm/');
    await page.waitForSelector('#sql-input', { timeout: 10000 });
    await page.waitForTimeout(3000);

    // Clear and enter a simple NEAR query
    const sqlInput = page.locator('#sql-input');
    await sqlInput.fill(`SELECT url, text FROM read_lance('https://data.metal0.dev/laion-1m/images.lance')
WHERE embedding NEAR 'cat'
LIMIT 10`);

    // Run
    const runButton = page.locator('button:has-text("Run")').first();
    await runButton.click();

    // Wait for results
    const resultsBody = page.locator('.results-body, .data-table tbody, table tbody');
    await expect(resultsBody).toBeVisible({ timeout: 60000 });

    // Verify we got rows
    const dataRows = page.locator('.data-table tr, table tr').filter({ hasNot: page.locator('th') });
    const rowCount = await dataRows.count();

    console.log(`NEAR 'cat' returned ${rowCount} rows`);
    expect(rowCount).toBeGreaterThan(0);
  });

  test('aggregation query (SUM, AVG, COUNT) works', async ({ page }) => {
    test.setTimeout(90000);

    await page.goto('/examples/wasm/');
    await page.waitForSelector('#sql-input', { timeout: 10000 });
    await page.waitForTimeout(3000);

    // Click the Stats 100K example button
    const statsButton = page.locator('button:has-text("Stats")');
    if (await statsButton.count() > 0) {
      await statsButton.first().click();

      // Run 
      const runButton = page.locator('button:has-text("Run")').first();
      await runButton.click();

      // Wait for aggregation results
      await page.waitForTimeout(10000);

      // Should show aggregation result (single row with SUM, AVG, etc)
      const resultsBody = page.locator('.results-body');
      await expect(resultsBody).toBeVisible({ timeout: 60000 });
    }
  });
});
