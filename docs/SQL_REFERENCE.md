# LanceQL SQL Reference

Complete reference for LanceQL's SQL dialect. LanceQL supports querying Lance, Parquet, Delta Lake, Iceberg, Arrow, Avro, ORC, and XLSX files.

## Table of Contents

- [Data Sources](#data-sources)
- [SELECT](#select)
- [WHERE](#where)
- [GROUP BY / HAVING](#group-by--having)
- [ORDER BY / LIMIT / OFFSET](#order-by--limit--offset)
- [JOINs](#joins)
- [Set Operations](#set-operations)
- [Window Functions](#window-functions)
- [Aggregate Functions](#aggregate-functions)
- [Scalar Functions](#scalar-functions)
- [Data Types](#data-types)
- [Operators](#operators)

---

## Data Sources

### read_lance()

Read Lance columnar files. Supports local files, URLs, and time travel.

```sql
-- Basic usage
SELECT * FROM read_lance('path/to/file.lance')

-- Remote URL
SELECT * FROM read_lance('https://example.com/data.lance')

-- Time travel (version number)
SELECT * FROM read_lance('path/to/file.lance', 24)

-- Browser: FILE is a special variable for uploaded files
SELECT * FROM read_lance(FILE) LIMIT 50
```

### read_parquet()

Read Parquet files.

```sql
SELECT * FROM read_parquet('path/to/file.parquet')
```

### read_delta()

Read Delta Lake tables.

```sql
SELECT * FROM read_delta('path/to/delta_table/')
```

### read_iceberg()

Read Apache Iceberg tables.

```sql
SELECT * FROM read_iceberg('path/to/iceberg_table/')
```

### read_arrow()

Read Arrow IPC files (.arrow, .arrows, .feather).

```sql
SELECT * FROM read_arrow('path/to/file.arrow')
```

### read_avro()

Read Apache Avro files.

```sql
SELECT * FROM read_avro('path/to/file.avro')
```

### read_orc()

Read Apache ORC files.

```sql
SELECT * FROM read_orc('path/to/file.orc')
```

### read_xlsx()

Read Microsoft Excel files.

```sql
-- Read first sheet
SELECT * FROM read_xlsx('path/to/file.xlsx')

-- Read specific sheet
SELECT * FROM read_xlsx('path/to/file.xlsx', 'Sheet2')
```

---

## SELECT

### Basic Syntax

```sql
SELECT column1, column2, ...
FROM data_source
```

### Column Selection

```sql
-- All columns
SELECT * FROM read_lance('data.lance')

-- Specific columns
SELECT id, name, created_at FROM read_lance('data.lance')

-- Column aliases
SELECT id AS user_id, name AS user_name FROM read_lance('data.lance')
```

### DISTINCT

Remove duplicate rows.

```sql
SELECT DISTINCT category FROM read_lance('products.lance')
```

### Expressions

```sql
-- Arithmetic
SELECT price * quantity AS total FROM read_lance('orders.lance')

-- String concatenation
SELECT first_name || ' ' || last_name AS full_name FROM read_lance('users.lance')

-- Function calls
SELECT UPPER(name), ROUND(price, 2) FROM read_lance('products.lance')
```

---

## WHERE

Filter rows based on conditions.

### Comparison Operators

```sql
SELECT * FROM read_lance('data.lance')
WHERE price > 100
  AND quantity <= 10
  AND status != 'cancelled'
```

### Logical Operators

```sql
-- AND, OR, NOT
SELECT * FROM read_lance('data.lance')
WHERE (price > 100 OR quantity > 50)
  AND NOT status = 'cancelled'
```

### IN / NOT IN

```sql
SELECT * FROM read_lance('data.lance')
WHERE category IN ('electronics', 'clothing', 'food')
```

### BETWEEN

```sql
SELECT * FROM read_lance('data.lance')
WHERE price BETWEEN 10 AND 100
```

### LIKE

Pattern matching with `%` (any characters) and `_` (single character).

```sql
SELECT * FROM read_lance('data.lance')
WHERE name LIKE 'John%'
  AND email LIKE '%@gmail.com'
```

### IS NULL / IS NOT NULL

```sql
SELECT * FROM read_lance('data.lance')
WHERE deleted_at IS NULL
```

---

## GROUP BY / HAVING

Aggregate rows by grouping columns.

```sql
SELECT category, COUNT(*), AVG(price)
FROM read_lance('products.lance')
GROUP BY category
HAVING COUNT(*) > 10
```

### Multiple Grouping Columns

```sql
SELECT category, region, SUM(sales)
FROM read_lance('sales.lance')
GROUP BY category, region
```

---

## ORDER BY / LIMIT / OFFSET

### ORDER BY

Sort results.

```sql
-- Ascending (default)
SELECT * FROM read_lance('data.lance') ORDER BY created_at

-- Descending
SELECT * FROM read_lance('data.lance') ORDER BY price DESC

-- Multiple columns
SELECT * FROM read_lance('data.lance') ORDER BY category ASC, price DESC
```

### LIMIT

Restrict number of rows returned.

```sql
SELECT * FROM read_lance('data.lance') LIMIT 100
```

### OFFSET

Skip rows before returning results.

```sql
-- Skip first 10 rows, return next 20
SELECT * FROM read_lance('data.lance') LIMIT 20 OFFSET 10
```

---

## JOINs

Combine rows from multiple tables.

### INNER JOIN

Returns rows with matches in both tables.

```sql
SELECT o.id, o.amount, c.name
FROM read_lance('orders.lance') o
INNER JOIN read_lance('customers.lance') c ON o.customer_id = c.id
```

### LEFT JOIN

Returns all rows from left table, matched rows from right.

```sql
SELECT o.id, o.amount, c.name
FROM read_lance('orders.lance') o
LEFT JOIN read_lance('customers.lance') c ON o.customer_id = c.id
```

### RIGHT JOIN

Returns all rows from right table, matched rows from left.

```sql
SELECT o.id, o.amount, c.name
FROM read_lance('orders.lance') o
RIGHT JOIN read_lance('customers.lance') c ON o.customer_id = c.id
```

### CROSS JOIN

Returns Cartesian product of both tables.

```sql
SELECT *
FROM read_lance('sizes.lance')
CROSS JOIN read_lance('colors.lance')
```

---

## Set Operations

### UNION

Combine results, removing duplicates.

```sql
SELECT name FROM read_lance('employees.lance')
UNION
SELECT name FROM read_lance('contractors.lance')
```

### UNION ALL

Combine results, keeping duplicates.

```sql
SELECT name FROM read_lance('employees.lance')
UNION ALL
SELECT name FROM read_lance('contractors.lance')
```

### INTERSECT

Return rows appearing in both queries.

```sql
SELECT email FROM read_lance('newsletter.lance')
INTERSECT
SELECT email FROM read_lance('customers.lance')
```

### EXCEPT

Return rows from first query not in second.

```sql
SELECT email FROM read_lance('all_users.lance')
EXCEPT
SELECT email FROM read_lance('unsubscribed.lance')
```

---

## Window Functions

Compute values across rows related to current row.

### Syntax

```sql
function() OVER (
    [PARTITION BY column1, column2, ...]
    [ORDER BY column1 [ASC|DESC], ...]
)
```

### ROW_NUMBER()

Sequential number within partition.

```sql
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM read_lance('employees.lance')
```

### RANK()

Rank with gaps for ties.

```sql
SELECT
    name,
    score,
    RANK() OVER (ORDER BY score DESC) AS rank
FROM read_lance('scores.lance')
```

### DENSE_RANK()

Rank without gaps for ties.

```sql
SELECT
    name,
    score,
    DENSE_RANK() OVER (ORDER BY score DESC) AS rank
FROM read_lance('scores.lance')
```

### LAG(column, offset, default)

Value from previous row in partition.

```sql
SELECT
    date,
    price,
    LAG(price, 1, 0) OVER (ORDER BY date) AS prev_price,
    price - LAG(price, 1, 0) OVER (ORDER BY date) AS change
FROM read_lance('prices.lance')
```

### LEAD(column, offset, default)

Value from following row in partition.

```sql
SELECT
    date,
    price,
    LEAD(price, 1, 0) OVER (ORDER BY date) AS next_price
FROM read_lance('prices.lance')
```

---

## Aggregate Functions

### COUNT

Count rows or non-null values.

```sql
SELECT COUNT(*) FROM read_lance('data.lance')                    -- All rows
SELECT COUNT(email) FROM read_lance('data.lance')                -- Non-null emails
SELECT COUNT(DISTINCT category) FROM read_lance('data.lance')    -- Unique categories
```

### SUM

Sum of values.

```sql
SELECT SUM(amount) FROM read_lance('orders.lance')
```

### AVG

Average of values.

```sql
SELECT AVG(price) FROM read_lance('products.lance')
```

### MIN / MAX

Minimum and maximum values.

```sql
SELECT MIN(created_at), MAX(created_at) FROM read_lance('events.lance')
```

### STDDEV / STDDEV_SAMP

Sample standard deviation.

```sql
SELECT STDDEV(score) FROM read_lance('results.lance')
```

### STDDEV_POP

Population standard deviation.

```sql
SELECT STDDEV_POP(score) FROM read_lance('results.lance')
```

### VARIANCE / VAR_SAMP

Sample variance.

```sql
SELECT VARIANCE(score) FROM read_lance('results.lance')
```

### VAR_POP

Population variance.

```sql
SELECT VAR_POP(score) FROM read_lance('results.lance')
```

### MEDIAN

Median value (50th percentile).

```sql
SELECT MEDIAN(price) FROM read_lance('products.lance')
```

### PERCENTILE / PERCENTILE_CONT / QUANTILE

Arbitrary percentile.

```sql
SELECT PERCENTILE(price, 0.95) FROM read_lance('products.lance')  -- 95th percentile
```

---

## Scalar Functions

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `UPPER(str)` | Convert to uppercase | `UPPER('hello')` → `'HELLO'` |
| `LOWER(str)` | Convert to lowercase | `LOWER('HELLO')` → `'hello'` |
| `LENGTH(str)` | String length | `LENGTH('hello')` → `5` |
| `TRIM(str)` | Remove leading/trailing whitespace | `TRIM('  hello  ')` → `'hello'` |

### Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `ABS(n)` | Absolute value | `ABS(-5)` → `5` |
| `ROUND(n, precision)` | Round to decimal places | `ROUND(3.14159, 2)` → `3.14` |
| `FLOOR(n)` | Round down | `FLOOR(3.7)` → `3` |
| `CEIL(n)` / `CEILING(n)` | Round up | `CEIL(3.2)` → `4` |

### Date/Time Functions

| Function | Description | Example |
|----------|-------------|---------|
| `YEAR(ts)` | Extract year | `YEAR(timestamp)` → `2024` |
| `MONTH(ts)` | Extract month (1-12) | `MONTH(timestamp)` → `6` |
| `DAY(ts)` | Extract day (1-31) | `DAY(timestamp)` → `15` |
| `HOUR(ts)` | Extract hour (0-23) | `HOUR(timestamp)` → `14` |
| `MINUTE(ts)` | Extract minute (0-59) | `MINUTE(timestamp)` → `30` |
| `SECOND(ts)` | Extract second (0-59) | `SECOND(timestamp)` → `45` |
| `DAYOFWEEK(ts)` / `DOW(ts)` | Day of week (0=Sun, 6=Sat) | `DAYOFWEEK(timestamp)` → `3` |
| `DAYOFYEAR(ts)` / `DOY(ts)` | Day of year (1-366) | `DAYOFYEAR(timestamp)` → `167` |
| `WEEK(ts)` | Week of year | `WEEK(timestamp)` → `24` |
| `QUARTER(ts)` | Quarter (1-4) | `QUARTER(timestamp)` → `2` |

### Date Operations

```sql
-- Extract date part
EXTRACT('year' FROM created_at)
DATE_PART('month', created_at)

-- Truncate to precision
DATE_TRUNC('month', created_at)   -- First day of month
DATE_TRUNC('year', created_at)    -- First day of year

-- Add interval
DATE_ADD(created_at, 7, 'day')    -- Add 7 days
DATEADD(created_at, 1, 'month')   -- Add 1 month

-- Difference between dates
DATE_DIFF(end_date, start_date, 'day')    -- Days between
DATEDIFF(end_date, start_date, 'hour')    -- Hours between

-- Unix timestamp conversion
EPOCH(created_at)                  -- Convert to Unix seconds
UNIX_TIMESTAMP(created_at)         -- Same as EPOCH
FROM_UNIXTIME(1704067200)          -- Convert from Unix seconds
TO_TIMESTAMP(1704067200)           -- Same as FROM_UNIXTIME
```

### Conditional Functions

| Function | Description | Example |
|----------|-------------|---------|
| `COALESCE(a, b, ...)` | First non-null value | `COALESCE(nickname, name)` |

### CASE Expression

```sql
SELECT
    name,
    CASE
        WHEN score >= 90 THEN 'A'
        WHEN score >= 80 THEN 'B'
        WHEN score >= 70 THEN 'C'
        ELSE 'F'
    END AS grade
FROM read_lance('students.lance')
```

---

## Data Types

LanceQL supports the following data types:

| Type | Description | Example Values |
|------|-------------|----------------|
| `int32` | 32-bit integer | `-2147483648` to `2147483647` |
| `int64` | 64-bit integer | `-9223372036854775808` to `9223372036854775807` |
| `float32` | 32-bit float | `3.14` |
| `float64` | 64-bit float | `3.141592653589793` |
| `bool` | Boolean | `true`, `false` |
| `string` / `utf8` | UTF-8 string | `'hello world'` |
| `timestamp[s]` | Timestamp (seconds) | Seconds since Unix epoch |
| `timestamp[ms]` | Timestamp (milliseconds) | Milliseconds since Unix epoch |
| `timestamp[us]` | Timestamp (microseconds) | Microseconds since Unix epoch |
| `timestamp[ns]` | Timestamp (nanoseconds) | Nanoseconds since Unix epoch |
| `date32` | Date (days) | Days since Unix epoch |
| `date64` | Date (milliseconds) | Milliseconds since Unix epoch |

---

## Operators

### Arithmetic

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `price + tax` |
| `-` | Subtraction | `total - discount` |
| `*` | Multiplication | `price * quantity` |
| `/` | Division | `total / count` |
| `%` | Modulo | `id % 10` |

### Comparison

| Operator | Description |
|----------|-------------|
| `=` | Equal |
| `!=` / `<>` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |

### Logical

| Operator | Description |
|----------|-------------|
| `AND` | Logical AND |
| `OR` | Logical OR |
| `NOT` | Logical NOT |

### String

| Operator | Description | Example |
|----------|-------------|---------|
| `\|\|` | Concatenation | `first_name \|\| ' ' \|\| last_name` |
| `LIKE` | Pattern match | `name LIKE 'John%'` |

### NULL

| Operator | Description |
|----------|-------------|
| `IS NULL` | Check for NULL |
| `IS NOT NULL` | Check for non-NULL |

### Set Membership

| Operator | Description |
|----------|-------------|
| `IN (...)` | Value in set |
| `NOT IN (...)` | Value not in set |
| `BETWEEN a AND b` | Value in range (inclusive) |

---

## Vector Search (NEAR Clause)

LanceQL supports semantic search with the `NEAR` clause. See [Vector Search Guide](./VECTOR_SEARCH.md) for details.

```sql
-- Text search
SELECT * FROM read_lance('images.lance') NEAR 'sunset beach' TOPK 20

-- Search specific column
SELECT * FROM read_lance('images.lance') NEAR embedding 'cat playing'

-- Combine with filters
SELECT * FROM read_lance('images.lance')
WHERE score > 0.5
NEAR 'mountain landscape' TOPK 50
```
