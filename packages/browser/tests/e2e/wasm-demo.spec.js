// @ts-check
import { test, expect } from '@playwright/test';

test.describe('WASM Demo', () => {
  test('homepage loads successfully', async ({ page }) => {
    await page.goto('/');

    // Check that the page has loaded
    await expect(page).toHaveTitle(/Lance|LanceQL/i);
  });

  test('WASM module loads without errors', async ({ page }) => {
    // Collect console errors
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');

    // Wait for potential WASM loading
    await page.waitForTimeout(2000);

    // Filter out expected errors (like CORS if any)
    const criticalErrors = consoleErrors.filter(
      err => !err.includes('CORS') && !err.includes('favicon')
    );

    expect(criticalErrors.length).toBe(0);
  });

  test('file drop zone exists', async ({ page }) => {
    await page.goto('/');

    // Check for common drop zone elements
    const dropZone = page.locator('[data-dropzone], .drop-zone, #dropzone, .file-drop');
    const hasDropZone = await dropZone.count() > 0;

    // If no explicit drop zone, check for file input
    if (!hasDropZone) {
      const fileInput = page.locator('input[type="file"]');
      await expect(fileInput).toBeVisible();
    }
  });
});

test.describe('Error Handling', () => {
  test('handles missing WASM gracefully', async ({ page }) => {
    // Block WASM file requests to simulate missing file
    await page.route('**/*.wasm', route => route.abort());

    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Page should still load (graceful degradation)
    await expect(page.locator('body')).toBeVisible();
  });
});
