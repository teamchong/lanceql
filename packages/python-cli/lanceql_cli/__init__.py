"""
LanceQL CLI - SQL for Lance/Parquet files

Usage:
    lanceql query data.lance "SELECT * FROM data LIMIT 10"
    lanceql ingest data.csv -o output.lance
    lanceql --help
"""

import os
import platform
import subprocess
import sys
import stat
import urllib.request
from pathlib import Path

__version__ = "0.2.0"

# GitHub release URL pattern
RELEASE_URL = "https://github.com/teamchong/lanceql/releases/download/v{version}/lanceql-{platform}"

PLATFORM_MAP = {
    ("Darwin", "arm64"): "darwin-arm64",
    ("Darwin", "x86_64"): "darwin-x64",
    ("Linux", "x86_64"): "linux-x64",
    ("Linux", "aarch64"): "linux-arm64",
    ("Windows", "AMD64"): "win32-x64.exe",
}


def get_binary_dir() -> Path:
    """Get the directory where the binary should be stored."""
    # Use ~/.local/bin on Unix, or AppData on Windows
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "lanceql" / "bin"
    else:
        return Path.home() / ".local" / "bin"


def get_binary_path() -> Path:
    """Get the path to the lanceql binary."""
    binary_dir = get_binary_dir()
    binary_name = "lanceql.exe" if platform.system() == "Windows" else "lanceql"
    return binary_dir / binary_name


def get_platform_key() -> str | None:
    """Get the platform key for downloading the correct binary."""
    system = platform.system()
    machine = platform.machine()
    return PLATFORM_MAP.get((system, machine))


def download_binary() -> Path:
    """Download the lanceql binary for the current platform."""
    platform_key = get_platform_key()
    if not platform_key:
        print(f"Unsupported platform: {platform.system()} {platform.machine()}", file=sys.stderr)
        print("Please build from source: https://github.com/teamchong/lanceql", file=sys.stderr)
        sys.exit(1)

    binary_path = get_binary_path()
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    url = RELEASE_URL.format(version=__version__, platform=platform_key)
    print(f"Downloading lanceql v{__version__} for {platform_key}...", file=sys.stderr)

    try:
        urllib.request.urlretrieve(url, binary_path)
        # Make executable on Unix
        if platform.system() != "Windows":
            binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Installed to {binary_path}", file=sys.stderr)
        return binary_path
    except Exception as e:
        print(f"Failed to download: {e}", file=sys.stderr)
        print(f"URL: {url}", file=sys.stderr)
        print("Please download manually or build from source.", file=sys.stderr)
        sys.exit(1)


def find_binary() -> Path:
    """Find or download the lanceql binary."""
    # Check if binary exists
    binary_path = get_binary_path()
    if binary_path.exists():
        return binary_path

    # Check system PATH
    import shutil
    system_binary = shutil.which("lanceql")
    if system_binary:
        return Path(system_binary)

    # Download if not found
    return download_binary()


def main():
    """Main entry point - execute the native binary."""
    binary_path = find_binary()

    try:
        result = subprocess.run(
            [str(binary_path)] + sys.argv[1:],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error executing lanceql: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
