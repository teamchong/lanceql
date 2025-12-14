"""Native ctypes bindings for lanceql shared library."""

import ctypes
from pathlib import Path
import sys
import numpy as np
from typing import Optional


# Platform-specific library name
if sys.platform == "darwin":
    lib_name = "liblanceql.dylib"
elif sys.platform == "win32":
    lib_name = "lanceql.dll"
else:
    lib_name = "liblanceql.so"

lib_path = Path(__file__).parent / lib_name
if not lib_path.exists():
    raise RuntimeError(f"Native library not found at {lib_path}")

_lib = ctypes.CDLL(str(lib_path))

# Function signatures
_lib.lance_open_memory.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.lance_open_memory.restype = ctypes.c_void_p

_lib.lance_close.argtypes = [ctypes.c_void_p]
_lib.lance_close.restype = None

_lib.lance_column_count.argtypes = [ctypes.c_void_p]
_lib.lance_column_count.restype = ctypes.c_uint32

_lib.lance_row_count.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.lance_row_count.restype = ctypes.c_uint64

_lib.lance_column_name.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.lance_column_name.restype = ctypes.c_size_t

_lib.lance_column_type.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.lance_column_type.restype = ctypes.c_size_t

_lib.lance_read_int64.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
]
_lib.lance_read_int64.restype = ctypes.c_size_t

_lib.lance_read_float64.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_lib.lance_read_float64.restype = ctypes.c_size_t

_lib.lance_read_string.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
]
_lib.lance_read_string.restype = ctypes.c_size_t


class LanceFile:
    """Python wrapper for native Lance file reader."""

    def __init__(self, data: bytes):
        """Open a Lance file from bytes.

        Args:
            data: Raw bytes of the Lance file

        Raises:
            TypeError: If data is not bytes
            ValueError: If file cannot be opened
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data)}")

        # Create C array
        c_data = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        self._handle = _lib.lance_open_memory(c_data, len(data))

        if not self._handle:
            raise ValueError("Failed to open Lance file")

    def close(self):
        """Close the file and free resources."""
        if self._handle:
            _lib.lance_close(self._handle)
            self._handle = None

    def column_count(self) -> int:
        """Get the number of columns in the table."""
        return _lib.lance_column_count(self._handle)

    def row_count(self, col_idx: int) -> int:
        """Get the number of rows for a specific column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            Number of rows in the column
        """
        return _lib.lance_row_count(self._handle, col_idx)

    def column_name(self, col_idx: int) -> str:
        """Get the name of a column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            Column name as string
        """
        buf = (ctypes.c_uint8 * 256)()
        length = _lib.lance_column_name(self._handle, col_idx, buf, 256)
        if length == 0:
            return ""
        return bytes(buf[:length]).decode("utf-8")

    def column_type(self, col_idx: int) -> str:
        """Get the logical type of a column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            Type string (e.g., "int64", "float64", "string")
        """
        buf = (ctypes.c_uint8 * 256)()
        length = _lib.lance_column_type(self._handle, col_idx, buf, 256)
        if length == 0:
            return ""
        return bytes(buf[:length]).decode("utf-8")

    def read_int64_column(self, col_idx: int) -> np.ndarray:
        """Read an int64 column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            NumPy array of int64 values
        """
        row_count = self.row_count(col_idx)
        if row_count == 0:
            return np.array([], dtype=np.int64)

        out = np.zeros(row_count, dtype=np.int64)
        actual = _lib.lance_read_int64(
            self._handle,
            col_idx,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            row_count,
        )
        return out[:actual]

    def read_float64_column(self, col_idx: int) -> np.ndarray:
        """Read a float64 column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            NumPy array of float64 values
        """
        row_count = self.row_count(col_idx)
        if row_count == 0:
            return np.array([], dtype=np.float64)

        out = np.zeros(row_count, dtype=np.float64)
        actual = _lib.lance_read_float64(
            self._handle,
            col_idx,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            row_count,
        )
        return out[:actual]

    def read_string_column(self, col_idx: int) -> list[str]:
        """Read a string column.

        Args:
            col_idx: Column index (0-based)

        Returns:
            List of strings
        """
        row_count = self.row_count(col_idx)
        if row_count == 0:
            return []

        # Allocate arrays for pointers and lengths
        string_ptrs = (ctypes.POINTER(ctypes.c_uint8) * row_count)()
        string_lens = (ctypes.c_size_t * row_count)()

        actual = _lib.lance_read_string(
            self._handle, col_idx, string_ptrs, string_lens, row_count
        )

        # Convert C strings to Python strings
        result = []
        for i in range(actual):
            # Create bytes from pointer and length, then decode UTF-8
            string_bytes = ctypes.string_at(string_ptrs[i], string_lens[i])
            result.append(string_bytes.decode("utf-8"))

        return result

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - ensure resources are freed."""
        self.close()
