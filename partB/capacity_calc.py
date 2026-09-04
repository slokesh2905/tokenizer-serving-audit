"""
Part B -- capacity reconciliation. Verifies by script the arithmetic that was
first worked out by hand (see NOTEBOOK.md), and checks it against bench_log.csv.
"""
import csv
import os

HERE = os.path.dirname(__file__)

# ---- from bench/model_spec.md ----
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_ELEM = 2  # fp16
GPU_BYTES_DECIMAL = 24 * 1_000_000_000  # 24 GB, decimal-GB reading of the spec
GPU_MEM_UTIL = 0.92
PARAMS = 4.2e9
WEIGHT_BYTES = PARAMS * BYTES_PER_ELEM
OVERHEAD_BYTES = 1.6 * 1_000_000_000
MAX_MODEL_LEN = 4096

print("=== B1: KV cache bytes/token & max concurrent sequences ===")
kv_bytes_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES_PER_ELEM
print(f"KV bytes/token = 2(K+V) x {LAYERS} layers x {KV_HEADS} kv_heads x {HEAD_DIM} head_dim x {BYTES_PER_ELEM} bytes")
print(f"               = {kv_bytes_per_token:,} bytes/token ({kv_bytes_per_token/1024:.1f} KiB/token)")

usable_for_kv_decimal = GPU_BYTES_DECIMAL * GPU_MEM_UTIL - WEIGHT_BYTES - OVERHEAD_BYTES
print(f"\nUsable mem for KV cache (decimal GB reading of '24GB'):")
print(f"  {GPU_BYTES_DECIMAL/1e9:.2f}GB x {GPU_MEM_UTIL} - weights({WEIGHT_BYTES/1e9:.2f}GB) - overhead({OVERHEAD_BYTES/1e9:.2f}GB)")
print(f"  = {usable_for_kv_decimal/1e9:.2f} GB usable for KV cache")

kv_per_seq_4096 = MAX_MODEL_LEN * kv_bytes_per_token
max_seqs_decimal = usable_for_kv_decimal / kv_per_seq_4096
print(f"\nKV cache per {MAX_MODEL_LEN}-token sequence = {kv_per_seq_4096:,} bytes ({kv_per_seq_4096/1e9:.3f} GB)")
print(f"Max concurrent {MAX_MODEL_LEN}-token sequences ~= {max_seqs_decimal:.1f}  ->  ~{int(max_seqs_decimal)} sequences")

# Sensitivity: same calc if "24 GB" is read as GiB (2^30) instead of decimal GB -- worth
# flagging as a caveat since it's a ~7% swing on the final number.
GPU_BYTES_GIB = 24 * (1024 ** 3)
usable_for_kv_gib = GPU_BYTES_GIB * GPU_MEM_UTIL - WEIGHT_BYTES - OVERHEAD_BYTES
max_seqs_gib = usable_for_kv_gib / kv_per_seq_4096
print(f"\n[caveat] If '24GB' / '4.2B params x 2 bytes' etc. are read as GiB/binary throughout instead:")
print(f"  max concurrent sequences ~= {max_seqs_gib:.1f} (vs {max_seqs_decimal:.1f} decimal) -- doesn't change the conclusion,"
      f" but the exact ceiling shifts")

print("\n=== Cross-check against bench_log.csv (prompt_len=3584 rows) ===")
rows = []
with open(os.path.join(HERE, "bench_log.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"{'batch':>6}{'prompt':>8}{'kv_util':>10}{'preempted':>11}{'reported_tok_s':>16}")
for r in rows:
    if r["prompt_len"] == "3584":
        print(f"{r['batch_size']:>6}{r['prompt_len']:>8}{r['kv_cache_util']:>10}{r['preempted_seqs']:>11}{r['reported_tok_s']:>16}")

print("\n=== B2: throughput anomaly ===")
long_rows = [r for r in rows if r["prompt_len"] == "3584"]
long_rows_sorted = sorted(long_rows, key=lambda r: int(r["batch_size"]))
peak = max(long_rows_sorted, key=lambda r: float(r["reported_tok_s"]))
print(f"Peak reported_tok_s in the prompt=3584 sweep is at batch={peak['batch_size']}: {peak['reported_tok_s']} tok/s")
for r in long_rows_sorted:
    if int(r["batch_size"]) > int(peak["batch_size"]):
        print(f"  batch={r['batch_size']}: reported_tok_s={r['reported_tok_s']} "
              f"(DROPS despite larger batch) | kv_cache_util={r['kv_cache_util']} preempted_seqs={r['preempted_seqs']}")

print("\n=== B3: reverse-engineering reported_tok_s, then honest goodput ===")
print("Hypothesis: reported_tok_s = (prompt_len + gen_len) * num_requests / wall_clock_s")
print(f"{'batch':>6}{'prompt':>8}{'gen':>6}{'n_req':>7}{'wall_s':>9}{'reported':>10}{'formula':>10}{'match':>8}")
all_match = True
for r in rows:
    p, g, n, w = int(r["prompt_len"]), int(r["gen_len"]), int(r["num_requests"]), float(r["wall_clock_s"])
    formula = (p + g) * n / w
    reported = float(r["reported_tok_s"])
    match = abs(formula - reported) < 0.5
    all_match = all_match and match
    print(f"{r['batch_size']:>6}{p:>8}{g:>6}{n:>7}{w:>9.2f}{reported:>10.1f}{formula:>10.1f}{'OK' if match else 'MISMATCH':>8}")
print("\nFORMULA CONFIRMED FOR ALL ROWS" if all_match else "FORMULA DOES NOT HOLD FOR ALL ROWS -- revise hypothesis")

print("\n--- Honest goodput for batch=24, prompt_len=3584 (two independent derivations) ---")
row24 = next(r for r in rows if r["batch_size"] == "24" and r["prompt_len"] == "3584")
gen_len = int(row24["gen_len"])
n_req = int(row24["num_requests"])
wall_s = float(row24["wall_clock_s"])
itl_ms = float(row24["itl_ms_p50"])
reported = float(row24["reported_tok_s"])

goodput_1 = gen_len * n_req / wall_s
per_seq_rate = 1000.0 / itl_ms
goodput_2 = per_seq_rate * n_req

print(f"Method 1 (generated tokens only / wall clock):  {gen_len} gen_len x {n_req} req / {wall_s}s = {goodput_1:.1f} tok/s")
print(f"Method 2 (from inter-token latency): 1000/{itl_ms}ms = {per_seq_rate:.2f} tok/s/seq  x {n_req} concurrent = {goodput_2:.1f} tok/s")
print(f"Reported (misleading) figure for this row: {reported:.1f} tok/s")
print(f"Overstatement factor: {reported/goodput_1:.1f}x (method 1) / {reported/goodput_2:.1f}x (method 2)")

print("\n=== B4 context: preemption counters already visible in the log ===")
for r in long_rows_sorted:
    print(f"  batch={r['batch_size']:>3}  kv_cache_util={r['kv_cache_util']}  preempted_seqs={r['preempted_seqs']}")
