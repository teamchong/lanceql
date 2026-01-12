import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 30000,
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
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
