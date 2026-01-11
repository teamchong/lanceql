/**
 * Shared test utilities for finding Lance fixture files.
 * Lance uses random hashes in filenames, so we need to find them dynamically.
 */

const path = require('path');
const fs = require('fs');

const FIXTURE_DIR = path.join(__dirname, '../../../tests/fixtures');
const BETTER_SQLITE3_DIR = path.join(FIXTURE_DIR, 'better-sqlite3');

/**
 * Find the .lance file in a dataset directory.
 * @param {string} datasetDir - Path to the dataset directory (e.g., 'simple_int64.lance')
 * @returns {string} Full path to the .lance file
 */
function findLanceFile(datasetDir) {
  const dataDir = path.join(datasetDir, 'data');
  if (!fs.existsSync(dataDir)) {
    throw new Error(`Data directory not found: ${dataDir}`);
  }
  const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.lance'));
  if (files.length === 0) {
    throw new Error(`No .lance files found in ${dataDir}`);
  }
  return path.join(dataDir, files[0]);
}

/**
 * Get path to a fixture Lance file.
 * @param {string} datasetName - Name of the dataset (e.g., 'simple_int64.lance')
 * @returns {string} Full path to the .lance file
 */
function getFixturePath(datasetName) {
  return findLanceFile(path.join(FIXTURE_DIR, datasetName));
}

/**
 * Get path to a better-sqlite3 fixture Lance file.
 * @param {string} datasetName - Name of the dataset (e.g., 'types_test.lance')
 * @returns {string} Full path to the .lance file
 */
function getBetterSqlite3FixturePath(datasetName) {
  return findLanceFile(path.join(BETTER_SQLITE3_DIR, datasetName));
}

module.exports = {
  FIXTURE_DIR,
  BETTER_SQLITE3_DIR,
  findLanceFile,
  getFixturePath,
  getBetterSqlite3FixturePath
};
