import { test, expect } from '@playwright/test';

test('SQL query with GROUP BY NEAR and LIKE filter - simple approach', async ({ page }) => {
  // Capture console logs
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('emoji') || text.includes('Error') || text.includes('Demo')) {
      console.log('BROWSER:', text);
    }
  });

  console.log('Navigating to WASM demo page...');
  await page.goto('/examples/wasm/', { waitUntil: 'networkidle' });

  // Wait for the emoji table to be created
  console.log('Waiting for emoji table...');
  await page.waitForSelector('[data-name="emoji"]', { timeout: 40000 });
  console.log('Emoji table found');

  // Wait for vector index AND for the auto-query to finish running
  // The page auto-runs a query 100ms after emoji table loads
  console.log('Waiting for auto-query to complete...');
  await page.waitForSelector('#results-table', { timeout: 15000 });
  console.log('Auto-query results appeared');

  // Additional wait to ensure everything is settled
  await page.waitForTimeout(2000);

  // Use JavaScript to set the SQL query directly and run it
  console.log('Setting SQL query and executing...');
  const results = await page.evaluate(async () => {
    const textarea = document.getElementById('sql-input');
    const runBtn = document.getElementById('run-sql-btn');

    if (!textarea || !runBtn) {
      return { error: 'Elements not found' };
    }

    // Set the query
    const query = "SELECT emoji, description FROM read_lance('opfs://emoji.lance') WHERE description LIKE '%sun%' GROUP BY NEAR description TOPK 3";
    textarea.value = query;

    // Trigger input event to update any listeners
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    // Click run button
    runBtn.click();

    // Wait for results with timeout
    await new Promise(resolve => setTimeout(resolve, 8000));

    // Get results
    const resultsTable = document.getElementById('results-table');
    if (!resultsTable) {
      const errorBar = document.querySelector('[style*="background: #dc3545"], [style*="background: #d32f2f"]');
      if (errorBar) {
        return { error: errorBar.textContent };
      }
      return { error: 'No results table found' };
    }

    const rows = Array.from(resultsTable.querySelectorAll('tbody tr'));
    const data = rows.map(row => {
      const cells = row.querySelectorAll('td');
      return {
        emoji: cells[0]?.textContent?.trim(),
        description: cells[1]?.textContent?.trim()
      };
    });

    const resultsCount = document.getElementById('results-count')?.textContent;

    return {
      rowCount: rows.length,
      resultsCount,
      data: data.slice(0, 10) // First 10 rows
    };
  });

  console.log('Results:', JSON.stringify(results, null, 2));

  if (results.error) {
    console.log('ERROR:', results.error);
  } else {
    console.log('Number of rows:', results.rowCount);
    console.log('Results count text:', results.resultsCount);
    console.log('Emojis:', results.data.map(r => r.emoji).join(', '));
  }

  // Take screenshot
  await page.screenshot({ path: '/tmp/emoji-query-simple.png', fullPage: true });
  console.log('Screenshot saved');
});