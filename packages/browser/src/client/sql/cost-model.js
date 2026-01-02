/**
 * CostModel - Query cost estimation for optimization
 * Extracted from query-planner.js for modularity
 */

class CostModel {
    constructor(options = {}) {
        this.isRemote = options.isRemote ?? true;

        // Network costs (ms)
        this.rttLatency = options.rttLatency ?? 50;
        this.bandwidthMBps = options.bandwidthMBps ?? 10;

        // CPU costs (ms per row)
        this.filterCostPerRow = options.filterCostPerRow ?? 0.001;
        this.hashBuildCostPerRow = options.hashBuildCostPerRow ?? 0.01;
        this.hashProbeCostPerRow = options.hashProbeCostPerRow ?? 0.005;

        // Memory costs
        this.memoryLimitMB = options.memoryLimitMB ?? 512;
    }

    estimateScanCost(rowCount, columnBytes, selectivity = 1.0) {
        const bytesToFetch = rowCount * columnBytes * selectivity;

        const networkCost = this.isRemote
            ? this.rttLatency + (bytesToFetch / (this.bandwidthMBps * 1024 * 1024)) * 1000
            : 0.1;

        const cpuCost = rowCount * this.filterCostPerRow;

        return {
            totalMs: networkCost + cpuCost,
            networkMs: networkCost,
            cpuMs: cpuCost,
            bytesToFetch,
            rowsToScan: rowCount * selectivity
        };
    }

    estimateJoinCost(leftRows, rightRows, leftBytes, rightBytes, joinSelectivity = 0.1) {
        const buildRows = Math.min(leftRows, rightRows);
        const buildBytes = buildRows < leftRows ? leftBytes : rightBytes;
        const buildCost = buildRows * this.hashBuildCostPerRow;

        const probeRows = Math.max(leftRows, rightRows);
        const probeCost = probeRows * this.hashProbeCostPerRow;

        const buildMemoryMB = (buildRows * buildBytes) / (1024 * 1024);
        const needsSpill = buildMemoryMB > this.memoryLimitMB;

        const spillCost = needsSpill ? buildMemoryMB * 10 : 0;

        return {
            totalMs: buildCost + probeCost + spillCost,
            buildMs: buildCost,
            probeMs: probeCost,
            spillMs: spillCost,
            needsSpill,
            outputRows: Math.round(leftRows * rightRows * joinSelectivity)
        };
    }

    estimateAggregateCost(inputRows, groupCount, aggCount) {
        const hashGroupCost = inputRows * this.hashBuildCostPerRow;
        const aggComputeCost = inputRows * aggCount * 0.0001;

        return {
            totalMs: hashGroupCost + aggComputeCost,
            outputRows: groupCount
        };
    }

    comparePlans(planA, planB) {
        const costA = planA.totalCost || this.estimatePlanCost(planA);
        const costB = planB.totalCost || this.estimatePlanCost(planB);

        return {
            recommended: costA.totalMs < costB.totalMs ? 'A' : 'B',
            costA,
            costB,
            savings: Math.abs(costA.totalMs - costB.totalMs)
        };
    }

    estimatePlanCost(plan) {
        let totalMs = 0;
        let totalBytes = 0;
        let operations = [];

        if (plan.leftScan) {
            const scanCost = this.estimateScanCost(
                plan.leftScan.estimatedRows || 10000,
                plan.leftScan.columnBytes || 100,
                plan.leftScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_left', ...scanCost });
        }

        if (plan.rightScan) {
            const scanCost = this.estimateScanCost(
                plan.rightScan.estimatedRows || 10000,
                plan.rightScan.columnBytes || 100,
                plan.rightScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_right', ...scanCost });
        }

        if (plan.join) {
            const joinCost = this.estimateJoinCost(
                plan.leftScan?.estimatedRows || 10000,
                plan.rightScan?.estimatedRows || 10000,
                plan.leftScan?.columnBytes || 100,
                plan.rightScan?.columnBytes || 100,
                plan.join.selectivity || 0.1
            );
            totalMs += joinCost.totalMs;
            operations.push({ op: 'join', ...joinCost });
        }

        if (plan.aggregations && plan.aggregations.length > 0) {
            const aggCost = this.estimateAggregateCost(
                plan.estimatedInputRows || 10000,
                plan.groupBy?.length || 1,
                plan.aggregations.length
            );
            totalMs += aggCost.totalMs;
            operations.push({ op: 'aggregate', ...aggCost });
        }

        return {
            totalMs,
            totalBytes,
            operations,
            isRemote: this.isRemote
        };
    }
}

export { CostModel };
