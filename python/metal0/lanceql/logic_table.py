"""Wrapper for compiled @logic_table shared libraries.

Provides a Pythonic interface to native @logic_table functions via ctypes.
Automatically discovers exported functions and sets up the FFI bindings.
"""

import ctypes
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Platform-specific library loading
if sys.platform == "darwin":
    _LOAD_LIB = lambda path: ctypes.CDLL(path)
elif sys.platform == "win32":
    _LOAD_LIB = lambda path: ctypes.CDLL(path, winmode=0)
else:
    _LOAD_LIB = lambda path: ctypes.CDLL(path)


class CompiledLogicTable:
    """Wrapper for a compiled @logic_table shared library.

    Provides Pythonic access to native @logic_table functions.
    Methods are discovered automatically from the shared library's
    exported symbols.

    Example:
        >>> from metal0.lanceql.logic_table import CompiledLogicTable
        >>> ops = CompiledLogicTable("/path/to/logic_table.dylib")
        >>> result = ops.VectorOps_dot_product([1.0, 2.0], [3.0, 4.0])

        # Or use attribute access for class-like interface:
        >>> result = ops.VectorOps.dot_product([1.0, 2.0], [3.0, 4.0])
    """

    def __init__(self, lib_path: Union[str, Path]):
        """Load a compiled @logic_table shared library.

        Args:
            lib_path: Path to the shared library (.so/.dylib/.dll)

        Raises:
            OSError: If library cannot be loaded
            ValueError: If library path doesn't exist
        """
        lib_path = Path(lib_path)
        if not lib_path.exists():
            raise ValueError(f"Library not found: {lib_path}")

        self._lib_path = str(lib_path)
        self._lib = _LOAD_LIB(self._lib_path)
        self._methods: Dict[str, Callable] = {}
        self._classes: Dict[str, "_ClassProxy"] = {}

        # Discover and bind exported functions
        self._discover_methods()

    def _discover_methods(self) -> None:
        """Discover and bind exported @logic_table functions.

        Exported functions follow the pattern: ClassName_methodName
        with C calling convention and specific argument types.
        """
        # Get all exported symbols
        symbols = self._get_exported_symbols()

        # Bind each function
        for symbol in symbols:
            # On macOS, C symbols have a leading underscore
            # Clean it to get the actual function name
            if symbol.startswith("_"):
                clean_symbol = symbol[1:]  # Remove leading underscore
            else:
                clean_symbol = symbol

            # Check for ClassName_methodName pattern
            if "_" in clean_symbol:
                # Filter out library internals (libdeflate, etc.)
                if not any(clean_symbol.startswith(prefix) for prefix in (
                    "libdeflate_", "std_", "runtime_", "c_interop_",
                )):
                    self._bind_method(symbol, clean_symbol)

    def _get_exported_symbols(self) -> List[str]:
        """Get list of exported symbols from the library."""
        import subprocess

        try:
            if sys.platform == "darwin":
                # Use nm on macOS
                result = subprocess.run(
                    ["nm", "-gU", self._lib_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    symbols = []
                    for line in result.stdout.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[1] == "T":
                            symbols.append(parts[2])
                    return symbols
            elif sys.platform == "win32":
                # Use dumpbin on Windows
                result = subprocess.run(
                    ["dumpbin", "/exports", self._lib_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    symbols = []
                    in_exports = False
                    for line in result.stdout.splitlines():
                        if "ordinal" in line.lower() and "name" in line.lower():
                            in_exports = True
                            continue
                        if in_exports and line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                symbols.append(parts[3])
                    return symbols
            else:
                # Use nm on Linux
                result = subprocess.run(
                    ["nm", "-gD", self._lib_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    symbols = []
                    for line in result.stdout.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[1] == "T":
                            symbols.append(parts[2])
                    return symbols
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Fallback: try common patterns
        return []

    def _bind_method(self, raw_symbol: str, clean_symbol: str) -> None:
        """Bind an exported function with appropriate type hints.

        The metal0 compiler exports functions with the pattern:
            ClassName_methodName(arg1, arg2, ..., len) -> f64

        For array arguments, a length parameter is added at the end.

        Args:
            raw_symbol: The raw symbol name from nm (with leading underscore on macOS)
            clean_symbol: The cleaned symbol name (without underscore)
        """
        try:
            func = getattr(self._lib, clean_symbol)
        except AttributeError:
            return  # Symbol not found

        # Parse the function name pattern: ClassName_methodName
        parts = clean_symbol.split("_", 1)
        if len(parts) != 2:
            return

        class_name, method_name = parts

        # Default signature: assumes (ptr, ptr, len) -> f64 for vector operations
        # This covers the common case of dot_product(a, b, len)
        # More complex signatures would need metadata from the compiler
        wrapper = self._create_wrapper(func, class_name, method_name)

        # Store the method
        self._methods[clean_symbol] = wrapper

        # Also register under class proxy
        if class_name not in self._classes:
            self._classes[class_name] = _ClassProxy()
        setattr(self._classes[class_name], method_name, wrapper)

    def _create_wrapper(
        self,
        func: ctypes._CFuncPtr,
        class_name: str,
        method_name: str,
    ) -> Callable:
        """Create a Python wrapper for a native function.

        The wrapper handles:
        - Converting Python lists to C arrays
        - Setting up argument types
        - Calling the function
        - Returning Python values
        """
        # Configure ctypes function signature
        # Assume the pattern: (f64*, f64*, usize) -> f64 for vector operations
        func.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # First array pointer
            ctypes.POINTER(ctypes.c_double),  # Second array pointer
            ctypes.c_size_t,                   # Length
        ]
        func.restype = ctypes.c_double

        def wrapper(a: Sequence[float], b: Sequence[float]) -> float:
            """Call the native function with Python sequences."""
            # Convert to C arrays
            arr_a = (ctypes.c_double * len(a))(*a)
            arr_b = (ctypes.c_double * len(b))(*b)

            # Call native function
            return func(arr_a, arr_b, len(a))

        wrapper.__name__ = method_name
        wrapper.__qualname__ = f"{class_name}.{method_name}"
        wrapper.__doc__ = f"Call native {class_name}.{method_name}"

        return wrapper

    def __getattr__(self, name: str) -> Any:
        """Provide attribute access to methods and class proxies.

        Allows:
        - ops.VectorOps_dot_product([...], [...]) - direct method call
        - ops.VectorOps.dot_product([...], [...]) - class-like access
        """
        # Check for direct method
        if name in self._methods:
            return self._methods[name]

        # Check for class proxy
        if name in self._classes:
            return self._classes[name]

        raise AttributeError(f"'{type(self).__name__}' has no method '{name}'")

    def __dir__(self) -> List[str]:
        """List available methods and classes."""
        return list(self._methods.keys()) + list(self._classes.keys())

    @property
    def methods(self) -> Dict[str, Callable]:
        """Get all available methods."""
        return dict(self._methods)

    @property
    def classes(self) -> Dict[str, "_ClassProxy"]:
        """Get all class proxies."""
        return dict(self._classes)


class _ClassProxy:
    """Proxy object for accessing methods under a class namespace.

    Allows: ops.VectorOps.dot_product([...], [...])
    instead of: ops.VectorOps_dot_product([...], [...])
    """

    def __init__(self):
        self._methods: Dict[str, Callable] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._methods[name] = value

    def __getattr__(self, name: str) -> Callable:
        if name in self._methods:
            return self._methods[name]
        raise AttributeError(f"Class has no method '{name}'")

    def __dir__(self) -> List[str]:
        return list(self._methods.keys())
