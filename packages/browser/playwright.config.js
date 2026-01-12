// @ts-check
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // Tests share WASM state - must run serially
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3100',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--use-gl=angle',
            '--use-angle=vulkan'
          ],
        },
      },
    },
    // Firefox temporarily disabled - browser binary missing
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
  ],
  webServer: {
    command: 'python3 -m http.server 3100 -d ../../',
    url: 'http://localhost:3100',
    reuseExistingServer: true,
    timeout: 30000,
  },
});
