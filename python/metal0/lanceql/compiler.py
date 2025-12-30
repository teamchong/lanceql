"""Runtime @logic_table compiler using metal0.

Compiles Python @logic_table classes to native shared libraries (.so/.dylib)
at runtime, enabling high-performance vectorized operations without
ahead-of-time compilation.

Example:
    >>> from metal0.lanceql import compile_logic_table
    >>>
    >>> source = '''
    ... from logic_table import logic_table
    ...
    ... @logic_table
    ... class VectorOps:
    ...     def dot_product(self, a: list, b: list) -> float:
    ...         result = 0.0
    ...         for i in range(len(a)):
    ...             result = result + a[i] * b[i]
    ...         return result
    ... '''
    >>>
    >>> ops = compile_logic_table(source)
    >>> result = ops.dot_product([1.0, 2.0], [3.0, 4.0])  # Returns 11.0
"""

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from .logic_table import CompiledLogicTable


class CompilerError(Exception):
    """Raised when @logic_table compilation fails."""
    pass


class LogicTableCompiler:
    """Compiles Python @logic_table classes to native shared libraries.

    Caches compiled modules by content hash to avoid recompilation.
    """

    def __init__(self, metal0_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize the compiler.

        Args:
            metal0_path: Path to metal0 binary. If None, searches in order:
                1. METAL0_PATH environment variable
                2. Bundled binary (if available)
                3. System PATH
            cache_dir: Directory for compiled modules. Defaults to ~/.cache/metal0/logic_table
        """
        self._metal0_path = self._find_metal0(metal0_path)
        self._cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _find_metal0(self, provided_path: Optional[str]) -> str:
        """Find the metal0 binary."""
        # 1. Use provided path
        if provided_path:
            if Path(provided_path).exists():
                return provided_path
            raise CompilerError(f"metal0 not found at provided path: {provided_path}")

        # 2. Check environment variable
        env_path = os.environ.get("METAL0_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # 3. Check bundled binary relative to this package
        # IMPORTANT: Must resolve() to handle paths with .. components (e.g., from sys.path manipulation)
        package_dir = Path(__file__).resolve().parent
        bundled_paths = [
            package_dir / "bin" / "metal0",  # Unix bundled
            package_dir / "bin" / "metal0.exe",  # Windows bundled
            package_dir.parent.parent.parent / "deps" / "metal0" / "zig-out" / "bin" / "metal0",  # Dev path
        ]
        for path in bundled_paths:
            if path.exists():
                return str(path)

        # 4. Check system PATH
        import shutil
        which_path = shutil.which("metal0")
        if which_path:
            return which_path

        raise CompilerError(
            "metal0 binary not found. Install via:\n"
            "  1. Set METAL0_PATH environment variable, or\n"
            "  2. Build from source: cd deps/metal0 && zig build, or\n"
            "  3. Add metal0 to PATH"
        )

    def _default_cache_dir(self) -> Path:
        """Get the default cache directory."""
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Caches" / "metal0" / "logic_table"
        elif sys.platform == "win32":
            return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "metal0" / "cache" / "logic_table"
        else:
            return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "metal0" / "logic_table"

    def _content_hash(self, source: str) -> str:
        """Compute content hash for caching."""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def _lib_extension(self) -> str:
        """Get platform-specific shared library extension."""
        if sys.platform == "darwin":
            return ".dylib"
        elif sys.platform == "win32":
            return ".dll"
        else:
            return ".so"

    def compile(self, source: str, force: bool = False) -> CompiledLogicTable:
        """Compile @logic_table source code to native module.

        Args:
            source: Python source code containing @logic_table decorated class(es)
            force: If True, recompile even if cached version exists

        Returns:
            CompiledLogicTable instance with callable methods

        Raises:
            CompilerError: If compilation fails
        """
        # Check cache
        content_hash = self._content_hash(source)
        lib_ext = self._lib_extension()
        cached_lib = self._cache_dir / f"logic_table_{content_hash}{lib_ext}"

        if not force and cached_lib.exists():
            return CompiledLogicTable(str(cached_lib))

        # Write source to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            temp_source = f.name

        try:
            # Compile using metal0
            # metal0 needs to run from its directory to find vendor dependencies
            # Binary is at deps/metal0/zig-out/bin/metal0, project root is deps/metal0
            metal0_bin = Path(self._metal0_path).resolve()
            # Go up from bin/metal0 -> zig-out -> metal0
            metal0_dir = metal0_bin.parent.parent.parent
            # Verify this is the right directory by checking for build.zig
            if not (metal0_dir / "build.zig").exists():
                # Try one level up (might be a different structure)
                metal0_dir = metal0_bin.parent.parent
                if not (metal0_dir / "build.zig").exists():
                    metal0_dir = None  # Give up, use current directory

            result = subprocess.run(
                [
                    str(metal0_bin),
                    "build",
                    "--emit-logic-table-shared",
                    temp_source,
                    "-o", str(cached_lib),
                ],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=str(metal0_dir) if metal0_dir else None,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilerError(f"Compilation failed:\n{error_msg}")

            if not cached_lib.exists():
                raise CompilerError(f"Compilation succeeded but output not found: {cached_lib}")

            # Verify the library is not empty (compilation may have failed silently)
            if cached_lib.stat().st_size == 0:
                # Check stderr/stdout for actual error messages
                error_msg = result.stderr or result.stdout
                if error_msg:
                    raise CompilerError(f"Compilation produced empty library:\n{error_msg}")
                raise CompilerError("Compilation produced empty library (no error message)")

            # Verify it's a valid shared library
            import struct
            with open(cached_lib, "rb") as f:
                magic = f.read(4)
                # Check for Mach-O (macOS), ELF (Linux), or PE (Windows)
                if magic[:4] not in (
                    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit
                    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit
                    b"\x7fELF",            # ELF
                    b"MZ",                 # PE (Windows)
                ):
                    error_msg = result.stderr or result.stdout
                    raise CompilerError(
                        f"Output is not a valid shared library (magic: {magic!r}):\n{error_msg}"
                    )

            return CompiledLogicTable(str(cached_lib))

        except subprocess.TimeoutExpired:
            raise CompilerError("Compilation timed out after 60 seconds")
        except subprocess.SubprocessError as e:
            raise CompilerError(f"Failed to run metal0: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_source)
            except OSError:
                pass

    def compile_file(self, file_path: Union[str, Path], force: bool = False) -> CompiledLogicTable:
        """Compile @logic_table from a Python file.

        Args:
            file_path: Path to Python source file
            force: If True, recompile even if cached version exists

        Returns:
            CompiledLogicTable instance with callable methods
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise CompilerError(f"Source file not found: {file_path}")

        source = file_path.read_text(encoding="utf-8")
        return self.compile(source, force=force)


# Module-level convenience functions

_default_compiler: Optional[LogicTableCompiler] = None


def _get_compiler() -> LogicTableCompiler:
    """Get or create the default compiler instance."""
    global _default_compiler
    if _default_compiler is None:
        _default_compiler = LogicTableCompiler()
    return _default_compiler


def compile_logic_table(source: str, force: bool = False) -> CompiledLogicTable:
    """Compile @logic_table source code to native module.

    This is a convenience function that uses a shared compiler instance.

    Args:
        source: Python source code containing @logic_table decorated class(es)
        force: If True, recompile even if cached version exists

    Returns:
        CompiledLogicTable instance with callable methods

    Example:
        >>> ops = compile_logic_table('''
        ... from logic_table import logic_table
        ...
        ... @logic_table
        ... class VectorOps:
        ...     def dot_product(self, a: list, b: list) -> float:
        ...         result = 0.0
        ...         for i in range(len(a)):
        ...             result = result + a[i] * b[i]
        ...         return result
        ... ''')
        >>> result = ops.dot_product([1.0, 2.0], [3.0, 4.0])  # Returns 11.0
    """
    return _get_compiler().compile(source, force=force)


def compile_logic_table_file(file_path: Union[str, Path], force: bool = False) -> CompiledLogicTable:
    """Compile @logic_table from a Python file.

    Args:
        file_path: Path to Python source file
        force: If True, recompile even if cached version exists

    Returns:
        CompiledLogicTable instance with callable methods
    """
    return _get_compiler().compile_file(file_path, force=force)
