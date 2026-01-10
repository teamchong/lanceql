# lanceql-cli

SQL for Lance/Parquet files - query columnar data from terminal.

## Installation

```bash
pip install lanceql-cli
# or
pipx install lanceql-cli
```

## Usage

```bash
# Query Lance files
lanceql query data.lance "SELECT * FROM data LIMIT 10"

# Query Parquet files
lanceql query data.parquet "SELECT name, value FROM data WHERE value > 100"

# Convert formats to Lance
lanceql ingest data.csv -o output.lance
lanceql ingest data.parquet -o output.lance
lanceql ingest data.json -o output.lance

# Vector search
lanceql query embeddings.lance "SELECT * FROM data NEAR 'search text' LIMIT 20"
```

## Supported Input Formats

- CSV, TSV, JSON, JSONL
- Parquet, Arrow/IPC, Avro, ORC
- Excel (XLSX/XLS)
- Delta Lake, Apache Iceberg

## Links

- [GitHub](https://github.com/teamchong/lanceql)
- [Documentation](https://github.com/teamchong/lanceql#readme)
- [Browser Package](https://www.npmjs.com/package/@metal0/lanceql)

## License

Apache-2.0
