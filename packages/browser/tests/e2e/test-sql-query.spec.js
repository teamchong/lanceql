import { test, expect } from '@playwright/test';

test('SQL query with GROUP BY NEAR and LIKE filter', async ({ page }) => {
  // Capture console logs
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('emoji') || text.includes('error') || text.includes('Error') || text.includes('Demo')) {
      console.log('BROWSER CONSOLE:', text);
    }
  });

  console.log('Navigating to WASM demo page...');
  await page.goto('/examples/wasm/');

  // Wait for page to load
  await page.waitForLoadState('domcontentloaded');

  // Wait for the SQL textarea to be visible
  await page.waitForSelector('#sql-input', { timeout: 10000 });

  // Wait for the emoji table to be auto-created (check if it appears in the tables list)
  console.log('Waiting for emoji table to be created...');
  await page.waitForSelector('[data-name="emoji"]', { timeout: 30000 });
  console.log('Emoji table found in tables list');

  // Wait for vector index creation to complete by monitoring console logs
  let vectorIndexCreated = false;
  page.on('console', msg => {
    if (msg.text().includes('Created vector index on emoji.description')) {
      vectorIndexCreated = true;
      console.log('Vector index creation confirmed');
    }
  });

  // Give it extra time to ensure the table is fully loaded and indexed
  console.log('Waiting for vector index to be created...');
  let attempts = 0;
  while (!vectorIndexCreated && attempts < 30) {
    await page.waitForTimeout(1000);
    attempts++;
  }

  if (vectorIndexCreated) {
    console.log('Vector index ready');
  } else {
    console.log('Warning: Vector index may not be ready yet');
  }

  // Additional wait to be safe
  await page.waitForTimeout(2000);

  // Uncheck "Emoji Table" checkbox if it's checked (to avoid auto-joining with LAION dataset)
  const emojiTableCheckbox = await page.locator('text="Emoji Table"');
  const isChecked = await emojiTableCheckbox.locator('..').locator('[type="checkbox"]').isChecked().catch(() => false);

  if (isChecked) {
    console.log('Unchecking Emoji Table checkbox...');
    await emojiTableCheckbox.click();
    await page.waitForTimeout(500);
  }

  // Fill SQL textarea
  const sqlQuery = "SELECT emoji, description FROM read_lance('opfs://emoji.lance') WHERE description LIKE '%sun%' GROUP BY NEAR description TOPK 3";
  console.log('Filling SQL textarea with query:', sqlQuery);

  // Clear existing content first
  await page.click('#sql-input', { clickCount: 3 }); // Triple-click to select all
  await page.keyboard.press('Delete');
  await page.fill('#sql-input', sqlQuery);

  // Click Run button
  console.log('Clicking Run button...');
  await page.click('#run-sql-btn');

  // Wait 5 seconds
  console.log('Waiting 5 seconds...');
  await page.waitForTimeout(5000);

  // Check for the error message bar at the bottom
  const errorBar = await page.locator('.alert-danger, [style*="background: #dc3545"], [style*="background: #d32f2f"]');
  const hasErrorBar = await errorBar.count() > 0;

  if (hasErrorBar) {
    const errorText = await errorBar.textContent();
    console.log('ERROR BAR:', errorText);
  }

  // Check for errors in the results notice
  const resultsNotice = await page.locator('#results-notice');
  const hasError = await resultsNotice.count() > 0;

  if (hasError) {
    const noticeText = await resultsNotice.textContent();
    const noticeClass = await resultsNotice.getAttribute('class');
    console.log('Results notice:', noticeText, 'Class:', noticeClass);
    if (noticeClass?.includes('error') || noticeClass?.includes('danger')) {
      console.log('ERROR FOUND:', noticeText);
    }
  }

  // Get results from the table
  const resultsTable = await page.locator('#results-table');
  const hasResults = await resultsTable.count() > 0;

  if (hasResults) {
    console.log('Results table found');

    // Count data rows (tbody tr)
    const rows = await page.locator('#results-table tbody tr').all();
    console.log('Number of rows:', rows.length);

    // Get emoji and description from each row
    const results = [];
    for (const row of rows) {
      const cells = await row.locator('td').all();
      if (cells.length >= 2) {
        const emoji = await cells[0].textContent();
        const description = await cells[1].textContent();
        results.push({ emoji: emoji?.trim(), description: description?.trim() });
      }
    }
    console.log('Results:', JSON.stringify(results, null, 2));
    console.log('Emojis returned:', results.map(r => r.emoji));
  } else {
    console.log('No results table found');
  }

  // Take a screenshot for inspection
  await page.screenshot({ path: '/tmp/sql-query-result.png', fullPage: true });
  console.log('Screenshot saved to /tmp/sql-query-result.png');
});
