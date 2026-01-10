// @ts-check
import { test, expect } from '@playwright/test';

// These tests verify the demo page works with bundled Alpine.js and nanostores
test.describe('Demo Page - Push-Down Filtering + JOIN + Vector Search', () => {

  /**
   * Wait for page to be ready (Alpine has rendered).
   * Don't use networkidle - worker keeps loading WASM.
   */
  async function waitForPageReady(page, timeout = 15000) {
    await page.waitForSelector('#sql-input', { timeout });
  }

  /**
   * Track network data transfers to the Lance dataset.
   * Returns a tracker object with getTotalBytes() method.
   */
  function trackDataTransfers(page) {
    let totalBytes = 0;
    let requestCount = 0;

    page.on('response', async (response) => {
      const url = response.url();
      // Only track actual data files (.lance) from our dataset
      if (url.includes('data.metal0.dev') && url.includes('/data/')) {
        const status = response.status();
        // 206 = Range request (partial content), 200 = full file
        if (status === 206 || status === 200) {
          const headers = response.headers();
          const contentLength = parseInt(headers['content-length'] || '0', 10);
          totalBytes += contentLength;
          requestCount++;
        }
      }
    });

    return {
      getTotalBytes: () => totalBytes,
      getTotalMB: () => totalBytes / (1024 * 1024),
      getRequestCount: () => requestCount
    };
  }

  test('page loads and Alpine initializes', async ({ page }) => {
    test.setTimeout(60000);

    page.on('console', msg => console.log('PAGE:', msg.text()));

    await page.goto('/examples/wasm/');

    await waitForPageReady(page);

    // Wait for SQL input to be visible
    const sqlInput = page.locator('#sql-input');
    await expect(sqlInput).toBeVisible({ timeout: 15000 });

    // Verify Run button exists
    const runButton = page.locator('#run-sql-btn');
    await expect(runButton).toBeVisible();

    console.log('Demo page loaded successfully with Alpine.js');
  });

  test('basic SELECT query works', async ({ page }) => {
    test.setTimeout(180000);

    page.on('console', msg => console.log('PAGE:', msg.text()));

    await page.goto('/examples/wasm/');

    await waitForPageReady(page);

    // Start tracking AFTER page load to exclude metadata fetches
    const tracker = trackDataTransfers(page);

    const sqlInput = page.locator('#sql-input');
    await expect(sqlInput).toBeVisible({ timeout: 15000 });

    // Simple SELECT without NEAR (no text embedding needed)
    await sqlInput.fill(`SELECT url, width, height, aesthetic FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 10`);

    await page.locator('#run-sql-btn').click();

    // Wait for results
    await page.waitForFunction(() => {
      const results = document.querySelector('.results-body');
      const status = document.querySelector('.status');
      return (results && results.children.length > 0) ||
             (status && status.textContent && status.textContent.includes('rows'));
    }, { timeout: 120000 });

    // Log download metrics - push-down filtering should ideally download < 5MB
    const downloadMB = tracker.getTotalMB();
    const requestCount = tracker.getRequestCount();
    console.log(`Basic SELECT query completed - downloaded ${downloadMB.toFixed(2)}MB in ${requestCount} requests`);
    // TODO: Enable once push-down filtering is fully implemented
    // expect(downloadMB).toBeLessThan(5);
  });

  test('aggregation query works', async ({ page }) => {
    test.setTimeout(180000);

    page.on('console', msg => console.log('PAGE:', msg.text()));

    await page.goto('/examples/wasm/');

    await waitForPageReady(page);

    // Start tracking AFTER page load to exclude metadata fetches
    const tracker = trackDataTransfers(page);

    const sqlInput = page.locator('#sql-input');
    await expect(sqlInput).toBeVisible({ timeout: 15000 });

    await sqlInput.fill(`SELECT SUM(aesthetic), AVG(aesthetic), COUNT(*) FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 1000`);

    await page.locator('#run-sql-btn').click();

    await page.waitForFunction(() => {
      const results = document.querySelector('.results-body');
      const status = document.querySelector('.status');
      return (results && results.children.length > 0) ||
             (status && status.textContent && status.textContent.includes('rows'));
    }, { timeout: 120000 });

    // Log download metrics - push-down filtering should ideally download < 10MB
    const downloadMB = tracker.getTotalMB();
    const requestCount = tracker.getRequestCount();
    console.log(`Aggregation query completed - downloaded ${downloadMB.toFixed(2)}MB in ${requestCount} requests`);
    // TODO: Enable once push-down filtering is fully implemented
    // expect(downloadMB).toBeLessThan(10);
  });
});
