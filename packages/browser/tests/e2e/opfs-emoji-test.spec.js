// @ts-check
import { test, expect } from '@playwright/test';

test.describe('OPFS emoji.lance SQL test', () => {
  test('should query emoji.lance from OPFS and return results', async ({ page }) => {
    // Navigate to the demo page
    await page.goto('http://localhost:3100/examples/wasm/', { waitUntil: 'networkidle' });

    // Wait 10 seconds as instructed
    await page.waitForTimeout(10000);

    // Find the SQL textarea and fill it with the query
    const sqlTextarea = page.locator('#sql-input');
    await sqlTextarea.fill("SELECT * FROM read_lance('opfs://emoji.lance') LIMIT 5");

    // Click the Run button
    const runButton = page.locator('#run-sql-btn');
    await runButton.click();

    // Wait 5 seconds as instructed
    await page.waitForTimeout(5000);

    // Capture console logs and errors
    const consoleLogs = [];
    const consoleErrors = [];

    page.on('console', msg => {
      consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    // Check for errors in the page
    const errorElements = await page.locator('.error, .error-message, [class*="error"]').all();
    const errors = [];
    for (const elem of errorElements) {
      const text = await elem.textContent();
      if (text && text.trim()) {
        errors.push(text.trim());
      }
    }

    // Try to find results table or output
    const resultsTable = page.locator('table, .results, [class*="result"]');
    const hasResults = await resultsTable.count() > 0;

    let rowCount = 0;
    let columns = [];
    let data = [];

    if (hasResults) {
      // Count rows (excluding header)
      const rows = resultsTable.locator('tr, .row, [class*="row"]');
      rowCount = await rows.count();

      // Get column headers
      const headers = resultsTable.locator('th, .column-header, [class*="header"]');
      const headerCount = await headers.count();
      for (let i = 0; i < headerCount; i++) {
        const headerText = await headers.nth(i).textContent();
        if (headerText) {
          columns.push(headerText.trim());
        }
      }

      // Get data from first few rows
      for (let i = 0; i < Math.min(5, rowCount); i++) {
        const cells = rows.nth(i).locator('td, .cell, [class*="cell"]');
        const cellCount = await cells.count();
        const rowData = [];
        for (let j = 0; j < cellCount; j++) {
          const cellText = await cells.nth(j).textContent();
          rowData.push(cellText ? cellText.trim() : '');
        }
        if (rowData.length > 0) {
          data.push(rowData);
        }
      }
    }

    // Report findings
    console.log('\n=== TEST REPORT ===');
    console.log('1. Errors:', errors.length > 0 ? errors.join(', ') : 'None');
    console.log('   Console errors:', consoleErrors.length > 0 ? consoleErrors.join(', ') : 'None');
    console.log('2. Number of rows returned:', rowCount > 0 ? rowCount - 1 : 0, '(excluding header)');
    console.log('3. Columns:', columns.length > 0 ? columns.join(', ') : 'Not found');
    console.log('   Data:', data.length > 0 ? JSON.stringify(data, null, 2) : 'No data captured');
    console.log('===================\n');

    // Take a screenshot for manual inspection
    await page.screenshot({ path: '/tmp/opfs-emoji-test.png', fullPage: true });
    console.log('Screenshot saved to: /tmp/opfs-emoji-test.png');

    // Get page HTML for debugging
    const pageContent = await page.content();
    const fs = await import('fs');
    fs.writeFileSync('/tmp/opfs-emoji-test.html', pageContent);
    console.log('Page HTML saved to: /tmp/opfs-emoji-test.html');

    // Basic assertions
    expect(errors.length).toBe(0);
    expect(rowCount).toBeGreaterThan(0);
  });
});
