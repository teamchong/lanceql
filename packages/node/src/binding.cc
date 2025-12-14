#include <napi.h>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#define RTLD_LAZY 0
static void* dlopen(const char* path, int) {
    return LoadLibraryA(path);
}
static void* dlsym(void* handle, const char* name) {
    return GetProcAddress((HMODULE)handle, name);
}
static int dlclose(void* handle) {
    return FreeLibrary((HMODULE)handle) ? 0 : 1;
}
static const char* dlerror() {
    static char buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(),
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, 256, NULL);
    return buf;
}
#else
#include <dlfcn.h>
#endif

// ============================================================================
// Zig Shared Library Loading
// ============================================================================

void* g_lib = nullptr;

// Platform-specific library extension
#ifdef _WIN32
#define LIB_EXT ".dll"
#define LIB_PREFIX ""
#elif __APPLE__
#define LIB_EXT ".dylib"
#define LIB_PREFIX "lib"
#else
#define LIB_EXT ".so"
#define LIB_PREFIX "lib"
#endif

// Get library path from environment or use defaults
static std::string getLibraryPath() {
    // 1. Check environment variable
    const char* env_path = std::getenv("LANCEQL_LIB_PATH");
    if (env_path && env_path[0]) {
        return std::string(env_path);
    }

    // 2. Try relative paths (for development)
    const char* rel_paths[] = {
        "../../zig-out/lib/" LIB_PREFIX "lanceql" LIB_EXT,  // From packages/node/
        "../zig-out/lib/" LIB_PREFIX "lanceql" LIB_EXT,     // From packages/
        "./lib/" LIB_PREFIX "lanceql" LIB_EXT,              // Installed in node_modules
        "./" LIB_PREFIX "lanceql" LIB_EXT,                  // Same directory
    };

    for (const char* path : rel_paths) {
        void* test = dlopen(path, RTLD_LAZY);
        if (test) {
            dlclose(test);
            return std::string(path);
        }
    }

    // 3. Fallback to system path (let dlopen search)
    return std::string(LIB_PREFIX "lanceql" LIB_EXT);
}

// Function pointer types
typedef void* (*lance_open_t)(const char*, size_t);
typedef void* (*lance_open_memory_t)(const uint8_t*, size_t);
typedef void (*lance_close_t)(void*);
typedef uint32_t (*lance_column_count_t)(void*);
typedef uint64_t (*lance_row_count_t)(void*, uint32_t);
typedef size_t (*lance_column_name_t)(void*, uint32_t, char*, size_t);
typedef size_t (*lance_column_type_t)(void*, uint32_t, char*, size_t);

typedef void* (*lance_sql_parse_t)(const char*, size_t, void*);
typedef void* (*lance_sql_execute_t)(void*, void*);
typedef void (*lance_sql_close_t)(void*);

typedef size_t (*lance_result_row_count_t)(void*);
typedef size_t (*lance_result_column_count_t)(void*);
typedef size_t (*lance_result_column_name_t)(void*, uint32_t, char*, size_t);
typedef uint32_t (*lance_result_column_type_t)(void*, uint32_t);
typedef size_t (*lance_result_read_int64_t)(void*, uint32_t, int64_t*, size_t);
typedef size_t (*lance_result_read_float64_t)(void*, uint32_t, double*, size_t);
typedef size_t (*lance_result_read_string_t)(void*, uint32_t, const char**, size_t*, size_t);
typedef void (*lance_result_close_t)(void*);
typedef void (*lance_cleanup_t)();

// Global function pointers
lance_open_t lance_open_fn = nullptr;
lance_open_memory_t lance_open_memory_fn = nullptr;
lance_close_t lance_close_fn = nullptr;
lance_column_count_t lance_column_count_fn = nullptr;
lance_row_count_t lance_row_count_fn = nullptr;
lance_column_name_t lance_column_name_fn = nullptr;
lance_column_type_t lance_column_type_fn = nullptr;

lance_sql_parse_t lance_sql_parse_fn = nullptr;
lance_sql_execute_t lance_sql_execute_fn = nullptr;
lance_sql_close_t lance_sql_close_fn = nullptr;

lance_result_row_count_t lance_result_row_count_fn = nullptr;
lance_result_column_count_t lance_result_column_count_fn = nullptr;
lance_result_column_name_t lance_result_column_name_fn = nullptr;
lance_result_column_type_t lance_result_column_type_fn = nullptr;
lance_result_read_int64_t lance_result_read_int64_fn = nullptr;
lance_result_read_float64_t lance_result_read_float64_fn = nullptr;
lance_result_read_string_t lance_result_read_string_fn = nullptr;
lance_result_close_t lance_result_close_fn = nullptr;
lance_cleanup_t lance_cleanup_fn = nullptr;

// ============================================================================
// Helper Structures
// ============================================================================

struct ColumnBuffer {
    std::string name;
    uint32_t type; // 0=int64, 1=float64, 2=string
    std::vector<int64_t> int64_data;
    std::vector<double> float64_data;
    std::vector<std::string> string_data;
};

// ============================================================================
// Statement Class
// ============================================================================

class Statement : public Napi::ObjectWrap<Statement> {
private:
    void* sql_handle;
    void* table_handle;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Statement(const Napi::CallbackInfo& info);
    ~Statement();

    Napi::Value All(const Napi::CallbackInfo& info);
    Napi::Value Get(const Napi::CallbackInfo& info);
    Napi::Value Run(const Napi::CallbackInfo& info);
};

Statement::Statement(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Statement>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
        return;
    }

    // Extract sql_handle and table_handle from External objects
    this->sql_handle = info[0].As<Napi::External<void>>().Data();
    this->table_handle = info[1].As<Napi::External<void>>().Data();
}

Statement::~Statement() {
    if (sql_handle && lance_sql_close_fn) {
        lance_sql_close_fn(sql_handle);
    }
}

Napi::Value Statement::All(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Execute query
    void* result_handle = lance_sql_execute_fn(this->sql_handle, this->table_handle);
    if (!result_handle) {
        Napi::Error::New(env, "Failed to execute SQL").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Get dimensions
    size_t row_count = lance_result_row_count_fn(result_handle);
    size_t col_count = lance_result_column_count_fn(result_handle);

    // Read all columns into buffers
    std::vector<ColumnBuffer> columns(col_count);
    for (size_t col_idx = 0; col_idx < col_count; col_idx++) {
        char name_buf[256];
        size_t name_len = lance_result_column_name_fn(result_handle, col_idx, name_buf, 256);
        columns[col_idx].name = std::string(name_buf, name_len);

        uint32_t col_type = lance_result_column_type_fn(result_handle, col_idx);
        columns[col_idx].type = col_type;

        if (col_type == 0) { // int64
            columns[col_idx].int64_data.resize(row_count);
            lance_result_read_int64_fn(result_handle, col_idx, columns[col_idx].int64_data.data(), row_count);
        } else if (col_type == 1) { // float64
            columns[col_idx].float64_data.resize(row_count);
            lance_result_read_float64_fn(result_handle, col_idx, columns[col_idx].float64_data.data(), row_count);
        } else if (col_type == 2) { // string
            // Allocate arrays for string pointers and lengths
            std::vector<const char*> str_ptrs(row_count);
            std::vector<size_t> str_lens(row_count);
            lance_result_read_string_fn(result_handle, col_idx, str_ptrs.data(), str_lens.data(), row_count);

            // Convert to std::string
            columns[col_idx].string_data.reserve(row_count);
            for (size_t i = 0; i < row_count; i++) {
                columns[col_idx].string_data.emplace_back(str_ptrs[i], str_lens[i]);
            }
        }
    }

    // Build row objects
    Napi::Array rows = Napi::Array::New(env, row_count);
    for (size_t row_idx = 0; row_idx < row_count; row_idx++) {
        Napi::Object row = Napi::Object::New(env);

        for (size_t col_idx = 0; col_idx < col_count; col_idx++) {
            const auto& col = columns[col_idx];
            if (col.type == 0) {
                row.Set(col.name, Napi::Number::New(env, col.int64_data[row_idx]));
            } else if (col.type == 1) {
                row.Set(col.name, Napi::Number::New(env, col.float64_data[row_idx]));
            } else if (col.type == 2) {
                row.Set(col.name, Napi::String::New(env, col.string_data[row_idx]));
            }
        }

        rows[row_idx] = row;
    }

    lance_result_close_fn(result_handle);
    return rows;
}

Napi::Value Statement::Get(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Get all rows, return first
    Napi::Value all_rows = this->All(info);
    if (!all_rows.IsArray()) {
        return env.Undefined();
    }

    Napi::Array rows = all_rows.As<Napi::Array>();
    if (rows.Length() == 0) {
        return env.Undefined();
    }

    return rows.Get(uint32_t(0));
}

Napi::Value Statement::Run(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // For read-only v0.1.0, execute query but return dummy result
    this->All(info);

    // Return {changes: 0, lastInsertRowid: 0}
    Napi::Object result = Napi::Object::New(env);
    result.Set("changes", Napi::Number::New(env, 0));
    result.Set("lastInsertRowid", Napi::Number::New(env, 0));
    return result;
}

Napi::Object Statement::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Statement", {
        InstanceMethod("all", &Statement::All),
        InstanceMethod("get", &Statement::Get),
        InstanceMethod("run", &Statement::Run),
    });

    exports.Set("Statement", func);
    return exports;
}

// ============================================================================
// Database Class
// ============================================================================

class Database : public Napi::ObjectWrap<Database> {
private:
    void* table_handle;
    bool is_open;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Database(const Napi::CallbackInfo& info);
    ~Database();

    Napi::Value Prepare(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);
};

Database::Database(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Database>(info), table_handle(nullptr), is_open(false) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Database path or buffer required").ThrowAsJavaScriptException();
        return;
    }

    // Check if input is a Buffer
    if (info[0].IsBuffer()) {
        Napi::Buffer<uint8_t> buffer = info[0].As<Napi::Buffer<uint8_t>>();
        table_handle = lance_open_memory_fn(buffer.Data(), buffer.Length());
    } else if (info[0].IsString()) {
        std::string path = info[0].As<Napi::String>().Utf8Value();

        // Validate input
        if (path == ":memory:" || path.empty()) {
            Napi::Error::New(env, "In-memory databases not supported").ThrowAsJavaScriptException();
            return;
        }

        table_handle = lance_open_fn(path.c_str(), path.length());
    } else {
        Napi::TypeError::New(env, "Expected string path or Buffer").ThrowAsJavaScriptException();
        return;
    }

    if (!table_handle) {
        Napi::Error::New(env, "Failed to open Lance file").ThrowAsJavaScriptException();
        return;
    }

    is_open = true;
}

Database::~Database() {
    if (table_handle && is_open && lance_close_fn) {
        lance_close_fn(table_handle);
        is_open = false;
    }
}

Napi::Value Database::Prepare(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!is_open) {
        Napi::Error::New(env, "Database connection is not open").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "SQL string expected").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string sql = info[0].As<Napi::String>().Utf8Value();

    void* sql_handle = lance_sql_parse_fn(sql.c_str(), sql.length(), this->table_handle);
    if (!sql_handle) {
        Napi::Error::New(env, "Failed to parse SQL").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Create Statement object using constructor from InstanceData
    auto* stmt_constructor = env.GetInstanceData<Napi::FunctionReference>();
    if (!stmt_constructor) {
        Napi::Error::New(env, "Statement constructor not initialized").ThrowAsJavaScriptException();
        return env.Null();
    }

    return stmt_constructor->New({
        Napi::External<void>::New(env, sql_handle),
        Napi::External<void>::New(env, this->table_handle)
    });
}

Napi::Value Database::Close(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (table_handle && is_open && lance_close_fn) {
        lance_close_fn(table_handle);
        is_open = false;
    }

    return env.Undefined();
}

Napi::Object Database::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Database", {
        InstanceMethod("prepare", &Database::Prepare),
        InstanceMethod("close", &Database::Close),
    });

    exports.Set("Database", func);
    return exports;
}

// ============================================================================
// Module Cleanup
// ============================================================================

static void CleanupModule(void* /*arg*/) {
    // Call Zig cleanup function to free all handles
    if (lance_cleanup_fn) {
        lance_cleanup_fn();
    }

    // Unload the shared library
    if (g_lib) {
        dlclose(g_lib);
        g_lib = nullptr;
    }
}

// ============================================================================
// Module Initialization
// ============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Load Zig shared library
    std::string lib_path = getLibraryPath();
    g_lib = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!g_lib) {
        Napi::Error::New(env, std::string("Failed to load library '") + lib_path + "': " + dlerror())
            .ThrowAsJavaScriptException();
        return exports;
    }

    // Load function pointers
    lance_open_fn = (lance_open_t)dlsym(g_lib, "lance_open");
    lance_open_memory_fn = (lance_open_memory_t)dlsym(g_lib, "lance_open_memory");
    lance_close_fn = (lance_close_t)dlsym(g_lib, "lance_close");
    lance_column_count_fn = (lance_column_count_t)dlsym(g_lib, "lance_column_count");
    lance_row_count_fn = (lance_row_count_t)dlsym(g_lib, "lance_row_count");
    lance_column_name_fn = (lance_column_name_t)dlsym(g_lib, "lance_column_name");
    lance_column_type_fn = (lance_column_type_t)dlsym(g_lib, "lance_column_type");

    lance_sql_parse_fn = (lance_sql_parse_t)dlsym(g_lib, "lance_sql_parse");
    lance_sql_execute_fn = (lance_sql_execute_t)dlsym(g_lib, "lance_sql_execute");
    lance_sql_close_fn = (lance_sql_close_t)dlsym(g_lib, "lance_sql_close");

    lance_result_row_count_fn = (lance_result_row_count_t)dlsym(g_lib, "lance_result_row_count");
    lance_result_column_count_fn = (lance_result_column_count_t)dlsym(g_lib, "lance_result_column_count");
    lance_result_column_name_fn = (lance_result_column_name_t)dlsym(g_lib, "lance_result_column_name");
    lance_result_column_type_fn = (lance_result_column_type_t)dlsym(g_lib, "lance_result_column_type");
    lance_result_read_int64_fn = (lance_result_read_int64_t)dlsym(g_lib, "lance_result_read_int64");
    lance_result_read_float64_fn = (lance_result_read_float64_t)dlsym(g_lib, "lance_result_read_float64");
    lance_result_read_string_fn = (lance_result_read_string_t)dlsym(g_lib, "lance_result_read_string");
    lance_result_close_fn = (lance_result_close_t)dlsym(g_lib, "lance_result_close");
    lance_cleanup_fn = (lance_cleanup_t)dlsym(g_lib, "lance_cleanup");

    // Check critical functions
    if (!lance_open_fn || !lance_sql_parse_fn || !lance_sql_execute_fn) {
        Napi::Error::New(env, "Failed to load required functions from library")
            .ThrowAsJavaScriptException();
        return exports;
    }

    // Initialize Statement first and store constructor in InstanceData
    Statement::Init(env, exports);

    // Store Statement constructor in InstanceData for Database::Prepare to use
    Napi::Function stmt_constructor = exports.Get("Statement").As<Napi::Function>();
    auto* constructor_ref = new Napi::FunctionReference();
    *constructor_ref = Napi::Persistent(stmt_constructor);
    env.SetInstanceData(constructor_ref);

    // Initialize Database
    Database::Init(env, exports);

    // Register cleanup hook
    napi_add_env_cleanup_hook(env, CleanupModule, nullptr);

    return exports;
}

NODE_API_MODULE(lanceql, Init)
