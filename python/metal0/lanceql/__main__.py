#!/usr/bin/env python3
"""
LanceQL CLI - Native query interface for Lance/Parquet files

Usage:
    lanceql read data.lance
    lanceql read data.parquet --columns id,name
    lanceql compile script.py -o output.so
    python -m metal0.lanceql read data.lance

Designed for high-performance data access with @logic_table compilation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def cmd_read(args):
    """Read Lance/Parquet file and print contents."""
    from . import parquet
    import pyarrow as pa

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    try:
        if file_path.endswith('.parquet'):
            table = parquet.read_table(file_path)
        elif file_path.endswith('.lance') or os.path.isdir(file_path):
            # Lance files are directories
            from ._native import LanceFile
            lf = LanceFile.from_path(file_path)
            table = lf.to_pyarrow()
        else:
            print(f"Error: Unknown file type: {file_path}", file=sys.stderr)
            return 1

        # Apply column filter
        if args.columns:
            cols = [c.strip() for c in args.columns.split(',')]
            table = table.select(cols)

        # Apply limit
        if args.limit:
            table = table.slice(0, args.limit)

        # Output format
        if args.json:
            print(json.dumps(table.to_pydict(), indent=2, default=str))
        elif args.csv:
            print(table.to_pandas().to_csv(index=False))
        else:
            print(table.to_pandas().to_string())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_compile(args):
    """Compile @logic_table Python file to native shared library."""
    from .compiler import compile_logic_table_file

    try:
        result = compile_logic_table_file(args.file, output=args.output, force=args.force)
        print(f"Compiled: {result.lib_path}")
        return 0
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return 1


def cmd_version(args):
    """Show version."""
    from . import __version__
    print(f"lanceql {__version__}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='lanceql',
        description='Native query interface for Lance/Parquet files'
    )
    parser.add_argument('--version', '-v', action='store_true', help='Show version')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # read command
    read_parser = subparsers.add_parser('read', help='Read Lance/Parquet file')
    read_parser.add_argument('file', help='File path')
    read_parser.add_argument('--columns', '-c', help='Comma-separated column names')
    read_parser.add_argument('--limit', '-n', type=int, help='Limit rows')
    read_parser.add_argument('--json', action='store_true', help='Output as JSON')
    read_parser.add_argument('--csv', action='store_true', help='Output as CSV')

    # compile command
    compile_parser = subparsers.add_parser('compile', help='Compile @logic_table to native')
    compile_parser.add_argument('file', help='Python file with @logic_table')
    compile_parser.add_argument('-o', '--output', help='Output path')
    compile_parser.add_argument('--force', '-f', action='store_true', help='Force recompile')

    args = parser.parse_args()

    if args.version:
        return cmd_version(args)

    if args.command == 'read':
        return cmd_read(args)
    elif args.command == 'compile':
        return cmd_compile(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
