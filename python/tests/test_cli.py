"""
LanceQL Python CLI Tests

Tests the CLI when run via pipx/uvx:
    pipx run metal0-lanceql --version
    pipx run metal0-lanceql read data.parquet
    python -m metal0.lanceql read data.lance
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import pytest

# Path to test fixtures
FIXTURES_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures"


def run_cli(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run the lanceql CLI via python -m."""
    cmd = [sys.executable, "-m", "metal0.lanceql"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    return result


def extract_json_from_output(output: str) -> str:
    """Extract JSON from output that may contain extra log lines."""
    lines = output.strip().split('\n')
    # Find where JSON object/array starts (first line that is ONLY { or [)
    # This avoids matching log lines like "[VectorAccelerator]..."
    json_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # JSON object or array typically starts with just { or [ on its own line
        # or { followed by content, but NOT [SomeText] which is a log prefix
        if stripped == '{' or stripped == '[':
            json_start = i
            break
        # Also match { "key": ... pattern
        if stripped.startswith('{') and ('"' in stripped or stripped == '{'):
            json_start = i
            break
    if json_start is not None:
        return '\n'.join(lines[json_start:])
    return output


class TestCLIVersionHelp:
    """Test version and help commands."""

    def test_version(self):
        """Test --version flag."""
        result = run_cli(["--version"], check=False)
        assert result.returncode == 0
        assert "lanceql" in result.stdout.lower()

    def test_help(self):
        """Test with no arguments shows help."""
        result = run_cli([], check=False)
        assert result.returncode == 0
        # Should show usage info
        assert "usage" in result.stdout.lower() or "lanceql" in result.stdout.lower()


class TestReadCommand:
    """Test the read command."""

    def test_read_parquet(self):
        """Test reading a Parquet file."""
        parquet_file = FIXTURES_PATH / "simple.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file)], check=False)
        assert result.returncode == 0
        # Should output some data
        assert len(result.stdout) > 0

    def test_read_parquet_json_output(self):
        """Test reading Parquet with JSON output."""
        parquet_file = FIXTURES_PATH / "simple.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file), "--json"], check=False)
        assert result.returncode == 0
        # Should be valid JSON (filter out any log lines before JSON)
        json_str = extract_json_from_output(result.stdout)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_read_parquet_csv_output(self):
        """Test reading Parquet with CSV output."""
        parquet_file = FIXTURES_PATH / "simple.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file), "--csv"], check=False)
        assert result.returncode == 0
        # Should have header row
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 1

    def test_read_parquet_limit(self):
        """Test reading with row limit."""
        parquet_file = FIXTURES_PATH / "simple.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file), "--limit", "2", "--json"], check=False)
        assert result.returncode == 0
        json_str = extract_json_from_output(result.stdout)
        data = json.loads(json_str)
        # Each column should have at most 2 values
        for col, values in data.items():
            assert len(values) <= 2

    def test_read_parquet_columns(self):
        """Test reading specific columns."""
        parquet_file = FIXTURES_PATH / "simple.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file), "--columns", "id", "--json"], check=False)
        assert result.returncode == 0
        json_str = extract_json_from_output(result.stdout)
        data = json.loads(json_str)
        # Should only have the 'id' column
        assert "id" in data
        assert len(data) == 1

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        result = run_cli(["read", "/nonexistent/file.parquet"], check=False)
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()

    def test_read_unknown_format(self):
        """Test reading a file with unknown format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"random data")
            temp_path = f.name

        try:
            result = run_cli(["read", temp_path], check=False)
            assert result.returncode != 0
        finally:
            os.unlink(temp_path)


class TestIngestViaCLI:
    """Test ingesting data via CLI (if supported)."""

    def test_cli_has_help(self):
        """Test that CLI provides help."""
        result = run_cli(["--help"], check=False)
        # Should not crash
        assert result.returncode == 0


class TestModuleExecution:
    """Test running as a module."""

    def test_python_m_execution(self):
        """Test python -m metal0.lanceql works."""
        result = subprocess.run(
            [sys.executable, "-m", "metal0.lanceql", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "lanceql" in result.stdout.lower()

    def test_direct_import(self):
        """Test that the module can be imported."""
        from metal0.lanceql import __main__
        assert hasattr(__main__, "main")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_command(self):
        """Test running with no command shows help."""
        result = run_cli([], check=False)
        assert result.returncode == 0
        # Should show help, not crash

    def test_invalid_command(self):
        """Test running with invalid command."""
        result = run_cli(["invalid_command"], check=False)
        # Should fail gracefully
        assert result.returncode != 0 or "error" in result.stderr.lower() or "invalid" in result.stdout.lower()


# Benchmark tests (optional, skip if fixtures missing)
class TestPerformance:
    """Performance sanity checks."""

    @pytest.mark.slow
    def test_read_large_file(self):
        """Test reading a larger file doesn't hang."""
        parquet_file = FIXTURES_PATH / "benchmark_100k.parquet"
        if not parquet_file.exists():
            pytest.skip(f"Fixture not found: {parquet_file}")

        result = run_cli(["read", str(parquet_file), "--limit", "10", "--json"], check=False)
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
