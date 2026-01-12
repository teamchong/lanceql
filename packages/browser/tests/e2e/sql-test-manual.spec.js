import { test, expect } from '@playwright/test';

test('SQL query with GROUP BY NEAR and LIKE filter', async ({ page }) => {
  // Capture console messages
  const consoleMessages = [];
  const errorMessages = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleMessages.push({ type: msg.type(), text });
    if (msg.type() === 'error') {
      errorMessages.push(text);
    }
  });

  // Navigate to the WASM example page
  await page.goto('/examples/wasm/');

  // Wait for page to be ready
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // Fill in the SQL textarea
  const sqlQuery = "SELECT emoji, description FROM read_lance('opfs://emoji.lance') WHERE description LIKE '%sun%' GROUP BY NEAR description TOPK 3";
  await page.fill('#sql-input', sqlQuery);

  // Click the Run button
  await page.click('#run-sql-btn');

  // Wait 5 seconds
  await page.waitForTimeout(5000);

  // Get result rows count
  const resultRows = await page.locator('table tbody tr').count();

  // Get status bar text
  const statusBar = await page.locator('#query-status').first();
  const statusText = await statusBar.textContent().catch(() => 'Status bar not found');

  // Print results
  console.log('\n===== TEST RESULTS =====');
  console.log(`1. Console Errors: ${errorMessages.length > 0 ? errorMessages.join('\n   ') : 'None'}`);
  console.log(`2. Result Rows: ${resultRows}`);
  console.log(`3. Status Bar: ${statusText}`);
  console.log('\n===== ALL CONSOLE MESSAGES =====');
  consoleMessages.forEach(msg => {
    console.log(`[${msg.type}] ${msg.text}`);
  });

  // Take a screenshot for reference
  await page.screenshot({ path: '/tmp/sql-test-result.png', fullPage: true });
  console.log('\nScreenshot saved to: /tmp/sql-test-result.png');
});
