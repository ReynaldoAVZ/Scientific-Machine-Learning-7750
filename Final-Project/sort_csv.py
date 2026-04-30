"""
sort_csv.py  —  Run this ONCE before the MATLAB analysis.
Sorts LongTest1.csv by (combo_id, trial_num) so MATLAB can read
trials sequentially without any row buffering.

Requirements: Python 3.8+, pandas  (pip install pandas)
Usage:        python sort_csv.py
Output:       LongTest1_sorted.csv in the same folder.

If you get a MemoryError the script automatically falls back to a
chunk-sort + k-way merge that uses O(chunk) memory.
"""

import pandas as pd
import os, sys, tempfile, heapq, csv

INPUT  = r'C:\Users\reyna\source\repos\Scientific-Machine-Learning-7750\Final-Project\LongTest1.csv'
OUTPUT = r'C:\Users\reyna\source\repos\Scientific-Machine-Learning-7750\Final-Project\LongTest1_sorted.csv'
CHUNKSIZE = 1_000_000   # lower if RAM is tight

print(f'Input : {INPUT}  ({os.path.getsize(INPUT)/1e9:.2f} GB)')
print(f'Output: {OUTPUT}')

try:
    print('Trying single-pass sort (needs ~2x file size in RAM)...')
    df = pd.read_csv(INPUT)
    df.sort_values(['combo_id', 'trial_num'], inplace=True)
    df.to_csv(OUTPUT, index=False)
    print(f'Done. {len(df):,} rows written to {OUTPUT}')

except MemoryError:
    print('Not enough RAM for single pass — using chunk sort + merge...')
    tmp_files = []
    for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNKSIZE)):
        chunk.sort_values(['combo_id', 'trial_num'], inplace=True)
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                          delete=False, newline='')
        chunk.to_csv(tmp, index=False)
        tmp.close()
        tmp_files.append(tmp.name)
        print(f'  Sorted chunk {i+1}  ({len(chunk):,} rows)')

    print(f'Merging {len(tmp_files)} chunks...')
    handles = [open(f, newline='') for f in tmp_files]
    readers = [csv.DictReader(h) for h in handles]
    fields  = readers[0].fieldnames

    with open(OUTPUT, 'w', newline='') as out_f:
        writer  = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()
        heap    = []
        for idx, r in enumerate(readers):
            row = next(r, None)
            if row:
                heapq.heappush(heap, (int(row['combo_id']),
                                      int(row['trial_num']), idx, row))
        written = 0
        while heap:
            cid, tid, idx, row = heapq.heappop(heap)
            writer.writerow(row)
            written += 1
            if written % 2_000_000 == 0:
                print(f'  Merged {written:,} rows...')
            nxt = next(readers[idx], None)
            if nxt:
                heapq.heappush(heap, (int(nxt['combo_id']),
                                      int(nxt['trial_num']), idx, nxt))

    for h in handles: h.close()
    for f in tmp_files: os.unlink(f)
    print(f'Done. {written:,} rows written to {OUTPUT}')