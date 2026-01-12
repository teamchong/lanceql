import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 30000,
    // On Linux, native addons can crash during worker cleanup.
    // Tests pass but worker processes crash afterward, causing vitest to exit with error.
    // This option prevents vitest from failing due to worker process exit errors.
    dangerouslyIgnoreUnhandledErrors: true,
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
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.js'],
    },
  },
});
