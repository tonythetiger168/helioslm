import json, glob, os

files = sorted(glob.glob('benchmarks/results_*.json'))
print('Found', len(files), 'result files')
print()

# Load all
all_data = []
for f in files:
    d = json.load(open(f))
    all_data.append((os.path.basename(f), d))

# Compare each with previous
for i in range(1, len(all_data)):
    name, curr = all_data[i]
    prev_name, prev = all_data[i-1]
    print(f'=== {prev_name} -> {name} ===')
    
    # Key metrics to compare
    metrics = [
        ('prefill tok/s', 'generation', 'prefill_tok_s'),
        ('decode tok/s', 'generation', 'decode_tok_s'),
        ('KV savings %', 'cache', 'kv_savings_pct'),
    ]
    
    for label, section, key in metrics:
        c = curr.get(section, {}).get(key, 'N/A')
        p = prev.get(section, {}).get(key, 'N/A')
        if c != 'N/A' and p != 'N/A':
            delta = c - p
            pct = (delta / p * 100) if p != 0 else 0
            status = 'OK' if abs(pct) < 5 else 'REGRESSION' if delta < 0 else 'IMPROVED'
            print(f'  {label:15s}: {p:.2f} -> {c:.2f} ({delta:+.2f}, {pct:+.1f}%) [{status}]')
        else:
            print(f'  {label:15s}: {p} -> {c}')
    
    print()

# Summary of latest
latest_name, latest = all_data[-1]
print(f'=== Latest: {latest_name} ===')
print(json.dumps(latest, indent=2)[:2000])
