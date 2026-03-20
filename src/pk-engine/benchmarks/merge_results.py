"""
Merge benchmark result CSV files from multiple CMD runs.

Usage:
    cd src/pk-engine
    python benchmarks/merge_results.py file1.csv [file2.csv ...]
"""

import csv
import sys
from pathlib import Path


def merge_csvs(input_files: list[str], output_file: str = "benchmark_results_merged.csv"):
    """Merge multiple CSV files with same headers into one."""
    all_rows = []
    fieldnames = None

    for fpath in input_files:
        p = Path(fpath)
        if not p.exists():
            print(f"[WARN] File not found: {fpath}")
            continue

        with open(p, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                if row.get('method', '').strip():  # skip empty rows
                    all_rows.append(row)
        print(f"  [OK] Read {p.name}: {len(all_rows)} total rows so far")

    if not all_rows or fieldnames is None:
        print("[ERR] No data to merge!")
        return

    out_path = Path(__file__).resolve().parent / output_file
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Merged {len(all_rows)} rows -> {out_path.name}")

    # ── Core Metrics Table ──
    print(f"\n{'='*100}")
    print(f"  CORE METRICS")
    print(f"{'='*100}")
    hdr = (f"{'Method':<16s} {'MPE':>7s} {'MAPE':>7s} {'RMSE':>7s} "
           f"{'CCC':>7s} {'Convg':>6s} {'Failed':>6s} {'Time':>9s} {'Speed':>8s}")
    print(hdr)
    print("-" * 100)
    for row in all_rows:
        m = row.get('method', '?')
        mpe = float(row.get('mpe', 0))
        mape = float(row.get('mape', 0))
        rmse = float(row.get('rmse', 0))
        ccc = float(row.get('ccc', 0))
        nc = row.get('n_converged', '?')
        nf = row.get('n_failed', '?')
        rt = float(row.get('runtime_s', 0))
        sp = float(row.get('speed_per_patient', 0))
        print(f"{m:<16s} {mpe:>+7.2f} {mape:>6.2f}% {rmse:>7.4f} "
              f"{ccc:>7.4f} {nc:>6s} {nf:>6s} {rt:>8.1f}s {sp:>7.2f}s")

    # ── Clinical Metrics Table ──
    print(f"\n{'='*100}")
    print(f"  CLINICAL METRICS")
    print(f"{'='*100}")
    hdr2 = (f"{'Method':<16s} {'BA_Bias':>8s} {'BA_LoA_L':>9s} {'BA_LoA_U':>9s} "
            f"{'Cov95%':>7s} {'TA%':>6s} {'Shrink%':>8s} {'TOST_p':>8s} {'TOST_Eq':>8s}")
    print(hdr2)
    print("-" * 100)
    for row in all_rows:
        m = row.get('method', '?')
        ba_b = float(row.get('ba_bias', 0))
        ba_l = float(row.get('ba_loa_lower', 0))
        ba_u = float(row.get('ba_loa_upper', 0))
        cov = float(row.get('coverage_95', 0))
        ta = float(row.get('target_attain', 0))
        sh = float(row.get('shrinkage_cl', 0))
        tp = float(row.get('tost_p', 0))
        te = row.get('tost_equivalent', '?')
        print(f"{m:<16s} {ba_b:>+8.4f} {ba_l:>+9.4f} {ba_u:>+9.4f} "
              f"{cov:>6.1f}% {ta:>5.1f}% {sh:>7.2f}% {tp:>8.4f} {te:>8s}")

    print(f"\n{'='*100}")
    print(f"  Output: {out_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_results.py file1.csv [file2.csv ...]")
        sys.exit(1)

    merge_csvs(sys.argv[1:])

