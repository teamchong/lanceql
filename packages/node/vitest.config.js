import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 30000,
    include: ['test/**/*.spec.js', 'test/**/*.test.js'],
    exclude: [
      'test/basic.test.js',
      'test/compat.test.js',
      'test/params.test.js',
      'test/distinct.test.js',
      'test/types.test.js',
      'test/timestamp.test.js',
      'test/logic-table-compiler.spec.js', // Requires metal0 build
    ],
    // Use threads instead of forks to avoid worker exit issues with native modules
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true, // Run tests sequentially to avoid native module conflicts
      },
    },
    // Ignore unhandled errors from worker cleanup (native module cleanup crashes)
    dangerouslyIgnoreUnhandledErrors: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.js'],
    },
  },
});
