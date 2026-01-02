class StatisticsManager {
    constructor() {
        this._cache = new Map();
        this._opfsRoot = null;
        this._computing = new Map();
    }

    async _getStatsDir() {
        if (this._opfsRoot) return this._opfsRoot;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            return null;
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._opfsRoot = await opfsRoot.getDirectoryHandle('lanceql-stats', { create: true });
            return this._opfsRoot;
        } catch {
            return null;
        }
    }

    _getCacheKey(datasetUrl) {
        let hash = 0;
        for (let i = 0; i < datasetUrl.length; i++) {
            hash = ((hash << 5) - hash) + datasetUrl.charCodeAt(i);
            hash |= 0;
        }
        return `stats_${Math.abs(hash).toString(16)}`;
    }

    async loadFromCache(datasetUrl, version) {
        const cacheKey = this._getCacheKey(datasetUrl);

        if (this._cache.has(cacheKey)) {
            const cached = this._cache.get(cacheKey);
            if (cached.version === version) {
                return cached;
            }
        }

        const statsDir = await this._getStatsDir();
        if (!statsDir) return null;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`);
            const file = await fileHandle.getFile();
            const text = await file.text();
            const cached = JSON.parse(text);

            if (cached.version !== version) {
                return null;
            }

            this._cache.set(cacheKey, cached);
            return cached;
        } catch (e) {
            return null;
        }
    }

    async saveToCache(datasetUrl, version, statistics) {
        const cacheKey = this._getCacheKey(datasetUrl);

        const cacheData = {
            datasetUrl,
            version,
            timestamp: Date.now(),
            columns: statistics.columns,
            fragments: statistics.fragments || null
        };

        this._cache.set(cacheKey, cacheData);

        const statsDir = await this._getStatsDir();
        if (!statsDir) return;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(JSON.stringify(cacheData));
            await writable.close();
        } catch {
            // Persist failure is non-fatal
        }
    }

    async getColumnStats(dataset, columnName, options = {}) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;
        const sampleSize = options.sampleSize || 100000;

        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.columns?.[columnName]) {
            return cached.columns[columnName];
        }

        const computeKey = `${datasetUrl}:${columnName}`;
        if (this._computing.has(computeKey)) {
            return this._computing.get(computeKey);
        }

        const computePromise = this._computeColumnStats(dataset, columnName, sampleSize);
        this._computing.set(computeKey, computePromise);

        try {
            const stats = await computePromise;

            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {} };
            existing.columns[columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } finally {
            this._computing.delete(computeKey);
        }
    }

    async _computeColumnStats(dataset, columnName, sampleSize) {
        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) {
            throw new Error(`Column not found: ${columnName}`);
        }

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        const stats = {
            column: columnName,
            type: colType,
            rowCount: 0,
            nullCount: 0,
            min: null,
            max: null,
            computed: true,
            sampleSize: 0
        };

        let rowsProcessed = 0;

        for (let fragIdx = 0; fragIdx < dataset._fragments.length && rowsProcessed < sampleSize; fragIdx++) {
            try {
                const fragFile = await dataset.openFragment(fragIdx);
                const fragRows = Math.min(
                    dataset._fragments[fragIdx].numRows,
                    sampleSize - rowsProcessed
                );

                const indices = Array.from({ length: fragRows }, (_, i) => i);
                const values = await fragFile.readColumnAtIndices(colIdx, indices);

                for (const value of values) {
                    stats.rowCount++;
                    stats.sampleSize++;

                    if (value === null || value === undefined) {
                        stats.nullCount++;
                        continue;
                    }

                    if (isNumeric) {
                        if (stats.min === null || value < stats.min) stats.min = value;
                        if (stats.max === null || value > stats.max) stats.max = value;
                    }
                }

                rowsProcessed += values.length;
            } catch {
                // Fragment read failure - continue with next
            }
        }

        return stats;
    }

    async precomputeForPlan(dataset, plan) {
        const filterColumns = new Set();

        for (const filter of (plan.pushedFilters || [])) {
            if (filter.column) filterColumns.add(filter.column);
            if (filter.left?.column) filterColumns.add(filter.left.column);
            if (filter.right?.column) filterColumns.add(filter.right.column);
        }

        const statsPromises = Array.from(filterColumns).map(col =>
            this.getColumnStats(dataset, col).catch(() => null)
        );

        const results = await Promise.all(statsPromises);
        const statsMap = new Map();

        Array.from(filterColumns).forEach((col, i) => {
            if (results[i]) statsMap.set(col, results[i]);
        });

        return statsMap;
    }

    canMatchFragment(fragmentStats, filter) {
        if (!fragmentStats || !filter) return true;

        const colStats = fragmentStats[filter.column];
        if (!colStats || colStats.min === null || colStats.max === null) return true;

        switch (filter.type) {
            case 'equality':
                return filter.value >= colStats.min && filter.value <= colStats.max;

            case 'range':
                switch (filter.op) {
                    case '>':
                        return colStats.max > filter.value;
                    case '>=':
                        return colStats.max >= filter.value;
                    case '<':
                        return colStats.min < filter.value;
                    case '<=':
                        return colStats.min <= filter.value;
                }
                break;

            case 'between':
                return colStats.max >= filter.low && colStats.min <= filter.high;

            case 'in':
                if (Array.isArray(filter.values)) {
                    return filter.values.some(v => v >= colStats.min && v <= colStats.max);
                }
                break;
        }

        return true;
    }

    async getFragmentStats(dataset, columnName, fragmentIndex) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;

        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.fragments?.[fragmentIndex]?.[columnName]) {
            return cached.fragments[fragmentIndex][columnName];
        }

        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) return null;

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        try {
            const fragFile = await dataset.openFragment(fragmentIndex);
            const fragRows = dataset._fragments[fragmentIndex].numRows;

            const sampleSize = Math.min(fragRows, 10000);
            const indices = Array.from({ length: sampleSize }, (_, i) => i);
            const values = await fragFile.readColumnAtIndices(colIdx, indices);

            const stats = {
                fragmentIndex,
                column: columnName,
                rowCount: fragRows,
                sampledRows: sampleSize,
                nullCount: 0,
                min: null,
                max: null
            };

            for (const value of values) {
                if (value === null || value === undefined) {
                    stats.nullCount++;
                    continue;
                }
                if (isNumeric) {
                    if (stats.min === null || value < stats.min) stats.min = value;
                    if (stats.max === null || value > stats.max) stats.max = value;
                }
            }

            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {}, fragments: {} };
            if (!existing.fragments) existing.fragments = {};
            if (!existing.fragments[fragmentIndex]) existing.fragments[fragmentIndex] = {};
            existing.fragments[fragmentIndex][columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } catch {
            return null;
        }
    }

    async getPrunableFragments(dataset, filters) {
        if (!filters || filters.length === 0 || !dataset._fragments) {
            return null;
        }

        const numFragments = dataset._fragments.length;
        const matchingFragments = [];
        let fragmentsPruned = 0;

        const filterColumns = new Set();
        for (const filter of filters) {
            if (filter.column) filterColumns.add(filter.column);
        }

        for (let fragIdx = 0; fragIdx < numFragments; fragIdx++) {
            let canPrune = false;

            for (const filter of filters) {
                if (!filter.column) continue;

                const fragStats = await this.getFragmentStats(dataset, filter.column, fragIdx);

                if (fragStats && !this.canMatchFragment({ [filter.column]: fragStats }, filter)) {
                    canPrune = true;
                    break;
                }
            }

            if (!canPrune) {
                matchingFragments.push(fragIdx);
            } else {
                fragmentsPruned++;
            }
        }

        return {
            matchingFragments,
            fragmentsPruned,
            totalFragments: numFragments
        };
    }
}

// Singleton instance
const statisticsManager = new StatisticsManager();

export { StatisticsManager, statisticsManager };
