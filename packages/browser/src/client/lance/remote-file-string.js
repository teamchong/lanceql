/**
 * RemoteLanceFile - String column readers
 * Extracted from remote-file.js for modularity
 */

import { parseStringColumnMeta } from './remote-file-proto.js';

/**
 * Read a single string at index via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number} rowIdx - Row index
 * @returns {Promise<string>}
 * @throws {Error} If the column is not a string column
 */
export async function readStringAt(file, colIdx, rowIdx) {
    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = parseStringColumnMeta(new Uint8Array(colMeta));

    // Check if this is actually a string column
    if (info.offsetsSize === 0 || info.dataSize === 0) {
        throw new Error(`Not a string column - offsetsSize=${info.offsetsSize}, dataSize=${info.dataSize}`);
    }

    // Calculate bytes per offset - strings have rows offsets of 4 or 8 bytes each
    const bytesPerOffset = info.offsetsSize / info.rows;

    // If bytesPerOffset is not 4 or 8, this is not a string column
    if (bytesPerOffset !== 4 && bytesPerOffset !== 8) {
        throw new Error(`Not a string column - bytesPerOffset=${bytesPerOffset}, expected 4 or 8`);
    }

    if (rowIdx >= info.rows) return '';

    // Determine offset size (4 or 8 bytes)
    const offsetSize = bytesPerOffset;

    // Fetch the two offsets for this string
    const offsetStart = info.offsetsStart + rowIdx * offsetSize;
    const offsetData = await file.fetchRange(offsetStart, offsetStart + offsetSize * 2 - 1);
    const offsetView = new DataView(offsetData);

    let strStart, strEnd;
    if (offsetSize === 4) {
        strStart = offsetView.getUint32(0, true);
        strEnd = offsetView.getUint32(4, true);
    } else {
        strStart = Number(offsetView.getBigUint64(0, true));
        strEnd = Number(offsetView.getBigUint64(8, true));
    }

    if (strEnd <= strStart) return '';
    const strLen = strEnd - strStart;

    // Fetch the string data
    const strData = await file.fetchRange(
        info.dataStart + strStart,
        info.dataStart + strEnd - 1
    );

    return new TextDecoder().decode(strData);
}

/**
 * Read multiple strings at indices via Range requests.
 * Uses batched fetching to minimize HTTP requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<string[]>}
 */
export async function readStringsAtIndices(file, colIdx, indices) {
    if (indices.length === 0) return [];

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = parseStringColumnMeta(new Uint8Array(colMeta));

    if (!info.pages || info.pages.length === 0) {
        return indices.map(() => '');
    }

    const results = new Array(indices.length).fill('');

    // Build page index with cumulative row counts
    let pageRowStart = 0;
    const pageIndex = [];
    for (const page of info.pages) {
        if (page.offsetsSize === 0 || page.dataSize === 0 || page.rows === 0) {
            pageRowStart += page.rows;
            continue;
        }
        pageIndex.push({
            start: pageRowStart,
            end: pageRowStart + page.rows,
            page
        });
        pageRowStart += page.rows;
    }

    // Group indices by page
    const pageGroups = new Map();
    for (let i = 0; i < indices.length; i++) {
        const rowIdx = indices[i];
        // Find which page contains this row
        for (let p = 0; p < pageIndex.length; p++) {
            const pi = pageIndex[p];
            if (rowIdx >= pi.start && rowIdx < pi.end) {
                if (!pageGroups.has(p)) {
                    pageGroups.set(p, []);
                }
                pageGroups.get(p).push({
                    globalIdx: rowIdx,
                    localIdx: rowIdx - pi.start,
                    resultIdx: i
                });
                break;
            }
        }
    }

    // Fetch strings from each page
    for (const [pageNum, items] of pageGroups) {
        const pi = pageIndex[pageNum];
        const page = pi.page;

        // Determine offset size (4 or 8 bytes per offset)
        const offsetSize = page.offsetsSize / page.rows;
        if (offsetSize !== 4 && offsetSize !== 8) continue;

        // Sort items by localIdx for efficient batching
        items.sort((a, b) => a.localIdx - b.localIdx);

        // Fetch offsets in batches
        const offsetBatches = [];
        let batchStart = 0;
        for (let i = 1; i <= items.length; i++) {
            if (i === items.length || items[i].localIdx - items[i-1].localIdx > 100) {
                offsetBatches.push(items.slice(batchStart, i));
                batchStart = i;
            }
        }

        // Collect string ranges from offset fetches
        // Lance string encoding: offset[N] = end of string N, start is offset[N-1] (or 0 if N=0)
        const stringRanges = [];

        await Promise.all(offsetBatches.map(async (batch) => {
            const minIdx = batch[0].localIdx;
            const maxIdx = batch[batch.length - 1].localIdx;

            // Fetch offsets: need offset[minIdx-1] through offset[maxIdx]
            const fetchStartIdx = minIdx > 0 ? minIdx - 1 : 0;
            const fetchEndIdx = maxIdx;
            const startOffset = page.offsetsStart + fetchStartIdx * offsetSize;
            const endOffset = page.offsetsStart + (fetchEndIdx + 1) * offsetSize - 1;
            const data = await file.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch) {
                // Position in fetched data
                const dataIdx = item.localIdx - fetchStartIdx;
                let strStart, strEnd;

                if (offsetSize === 4) {
                    strEnd = view.getUint32(dataIdx * 4, true);
                    strStart = item.localIdx === 0 ? 0 : view.getUint32((dataIdx - 1) * 4, true);
                } else {
                    strEnd = Number(view.getBigUint64(dataIdx * 8, true));
                    strStart = item.localIdx === 0 ? 0 : Number(view.getBigUint64((dataIdx - 1) * 8, true));
                }

                if (strEnd > strStart) {
                    stringRanges.push({
                        start: strStart,
                        end: strEnd,
                        resultIdx: item.resultIdx,
                        dataStart: page.dataStart
                    });
                }
            }
        }));

        // Fetch string data
        if (stringRanges.length > 0) {
            stringRanges.sort((a, b) => a.start - b.start);

            // Batch nearby string fetches
            const dataBatches = [];
            let dbStart = 0;
            for (let i = 1; i <= stringRanges.length; i++) {
                if (i === stringRanges.length ||
                    stringRanges[i].start - stringRanges[i-1].end > 4096) {
                    dataBatches.push({
                        rangeStart: stringRanges[dbStart].start,
                        rangeEnd: stringRanges[i-1].end,
                        items: stringRanges.slice(dbStart, i),
                        dataStart: stringRanges[dbStart].dataStart
                    });
                    dbStart = i;
                }
            }

            await Promise.all(dataBatches.map(async (batch) => {
                const data = await file.fetchRange(
                    batch.dataStart + batch.rangeStart,
                    batch.dataStart + batch.rangeEnd - 1
                );
                const bytes = new Uint8Array(data);

                for (const item of batch.items) {
                    const localStart = item.start - batch.rangeStart;
                    const len = item.end - item.start;
                    const strBytes = bytes.slice(localStart, localStart + len);
                    results[item.resultIdx] = new TextDecoder().decode(strBytes);
                }
            }));
        }
    }

    return results;
}
