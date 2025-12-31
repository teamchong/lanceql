#!/usr/bin/env node
/**
 * Postinstall script for @metal0/lanceql
 *
 * Verifies the binary is available for the current platform.
 * Platform-specific binaries are distributed via optional dependencies.
 */

const fs = require('fs');
const path = require('path');

const PLATFORM_PACKAGES = {
  'darwin-arm64': '@metal0/lanceql-darwin-arm64',
  'darwin-x64': '@metal0/lanceql-darwin-x64',
  'linux-arm64': '@metal0/lanceql-linux-arm64',
  'linux-x64': '@metal0/lanceql-linux-x64',
  'win32-x64': '@metal0/lanceql-win32-x64',
};

function checkBinary() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;

  const platformPkg = PLATFORM_PACKAGES[key];
  if (!platformPkg) {
    console.warn(`Warning: No pre-built binary for ${key}`);
    console.warn('You may need to build from source: zig build cli');
    return;
  }

  // Check if platform package was installed
  try {
    const pkgPath = require.resolve(`${platformPkg}/package.json`);
    const pkgDir = path.dirname(pkgPath);
    const binName = platform === 'win32' ? 'lanceql.exe' : 'lanceql';
    const binPath = path.join(pkgDir, 'bin', binName);

    if (fs.existsSync(binPath)) {
      console.log(`lanceql binary installed for ${key}`);
      return;
    }
  } catch (e) {
    // Package not found
  }

  // Check local build
  const localBin = path.join(__dirname, '..', '..', '..', 'zig-out', 'bin', 'lanceql');
  if (fs.existsSync(localBin)) {
    console.log('Using locally built lanceql binary');
    return;
  }

  console.warn(`Warning: lanceql binary not found for ${key}`);
  console.warn('Install options:');
  console.warn(`  1. npm install ${platformPkg}`);
  console.warn('  2. Build from source: zig build cli');
}

checkBinary();
