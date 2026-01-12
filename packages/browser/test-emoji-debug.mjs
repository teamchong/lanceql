import { chromium } from '@playwright/test';

(async () => {
  try {
    console.log('Connecting to CDP on port 9222...');
    const browser = await chromium.connectOverCDP('http://localhost:9222');
    const contexts = browser.contexts();
    const context = contexts[0];
    const pages = await context.pages();
    let page = pages.length > 0 ? pages[0] : await context.newPage();

    console.log('Navigating to http://localhost:3100/examples/wasm/...');

    // Capture console logs from the start
    const consoleLogs = [];
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      console.log(`[Browser Console] ${text}`);
    });

    await page.goto('http://localhost:3100/examples/wasm/', { waitUntil: 'load', timeout: 30000 });

    console.log('\n=== CLEARING EMOJI.LANCE FROM OPFS ===');
    // Delete emoji.lance to force recreation
    await page.evaluate(async () => {
      try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry('emoji.lance', { recursive: true });
        console.log('[Test] Deleted emoji.lance from OPFS');
      } catch (e) {
        console.log('[Test] No emoji.lance to delete or error:', e.message);
      }
    });

    console.log('Reloading page with cache disabled...');
    await page.reload({ waitUntil: 'load' });

    console.log('Waiting 10 seconds for page to fully load and auto-create emoji.lance...');
    await page.waitForTimeout(10000);

    // Check for recreation messages
    const hasRecreationLog = consoleLogs.some(log =>
      log.includes('emoji.lance is corrupted') ||
      log.includes('Auto-creating emoji.lance') ||
      log.includes('Created emoji.lance')
    );

    console.log('\n=== FILLING SQL QUERY ===');
    const sqlQuery = `SELECT emoji, description FROM read_lance('opfs://emoji.lance') WHERE description LIKE '%sun%' GROUP BY NEAR description TOPK 3`;

    const textarea = await page.locator('textarea').first();
    await textarea.fill(sqlQuery);
    console.log('Query filled:', sqlQuery);

    console.log('\n=== CLICKING RUN BUTTON ===');
    const runButton = await page.locator('button:has-text("Run")').first();
    await runButton.click();

    console.log('Waiting 5 seconds for query execution...');
    await page.waitForTimeout(5000);

    // Check for errors
    const errorElement = await page.locator('.error, .alert-error, [role="alert"]').first();
    const hasError = await errorElement.count() > 0;
    let errorText = '';
    if (hasError) {
      errorText = await errorElement.textContent();
    }

    // Count result rows
    const resultRows = await page.locator('table tbody tr');
    const rowCount = await resultRows.count();

    console.log('\n=== RESULTS ===');
    console.log('1. Emoji table recreated?', hasRecreationLog ? 'YES' : 'NO (no recreation message found)');
    console.log('2. Any errors?', hasError ? `YES: ${errorText}` : 'NO');
    console.log('3. Result rows:', rowCount);

    // Get table content if exists
    if (rowCount > 0) {
      console.log('\n=== TABLE CONTENT ===');
      for (let i = 0; i < Math.min(rowCount, 10); i++) {
        const row = resultRows.nth(i);
        const cells = await row.locator('td');
        const cellCount = await cells.count();
        const cellTexts = [];
        for (let j = 0; j < cellCount; j++) {
          cellTexts.push(await cells.nth(j).textContent());
        }
        console.log(`Row ${i + 1}:`, cellTexts);
      }
    }

    await browser.close();
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
})();
