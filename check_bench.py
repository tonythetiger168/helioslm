import json, glob, os
files = sorted(glob.glob('benchmarks/results_*.json'))
print('Found', len(files), 'result files')
for f in files:
    d = json.load(open(f))
    print(os.path.basename(f), '->', list(d.keys())[:5])
