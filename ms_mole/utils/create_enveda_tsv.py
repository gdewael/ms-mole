"""
Convert enveda MGF (gzipped) to a TSV matching the MassSpecGym format expected
by MsMoleMassSpecDataModule.

Folds are read from a pre-existing split TSV (columns: id, fold) keyed by
spectrum TITLE. Peaks are filtered to the top --max_peaks by intensity before
normalization.

Usage:
    python create_enveda_tsv.py \
        /path/to/enveda-180.mgf.gz \
        /path/to/split_enveda.tsv \
        /path/to/enveda.tsv \
        [--max_peaks 128]
"""

import gzip
import argparse
import csv


def parse_mgf_gz(mgf_gz_path, max_peaks):
    """Yield dicts of spectrum data from a gzipped MGF file."""
    with gzip.open(mgf_gz_path, "rt") as f:
        current_meta = {}
        peaks_mz = []
        peaks_int = []

        for line in f:
            line = line.rstrip()
            if line == "BEGIN IONS":
                current_meta = {}
                peaks_mz = []
                peaks_int = []
            elif line == "END IONS":
                if peaks_mz:
                    # Keep only top max_peaks by intensity
                    if len(peaks_mz) > max_peaks:
                        order = sorted(
                            range(len(peaks_int)),
                            key=lambda i: peaks_int[i],
                            reverse=True,
                        )[:max_peaks]
                        order_sorted = sorted(order)  # restore m/z order
                        peaks_mz = [peaks_mz[i] for i in order_sorted]
                        peaks_int = [peaks_int[i] for i in order_sorted]

                    max_int = max(peaks_int)
                    current_meta["_mzs"] = peaks_mz
                    current_meta["_intensities"] = [v / max_int for v in peaks_int]
                    yield current_meta
            elif "=" in line and not line[0].isdigit():
                key, _, val = line.partition("=")
                current_meta[key] = val
            elif line and (line[0].isdigit() or line[0] == "."):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks_mz.append(float(parts[0]))
                        peaks_int.append(float(parts[1]))
                    except ValueError:
                        pass


def load_split(split_tsv_path):
    """Return dict mapping spectrum id (TITLE) -> fold."""
    fold_map = {}
    with open(split_tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fold_map[row["id"]] = row["fold"]
    return fold_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert enveda MGF.gz to MassSpecGym-format TSV"
    )
    parser.add_argument("mgf_gz_path", type=str, help="Path to enveda-*.mgf.gz")
    parser.add_argument("split_tsv_path", type=str, help="Path to split_enveda.tsv (id, fold)")
    parser.add_argument("out_tsv_path", type=str, help="Output .tsv path")
    parser.add_argument("--max_peaks", type=int, default=128,
                        help="Keep top N peaks by intensity (default: 128)")
    args = parser.parse_args()

    print(f"Loading split from {args.split_tsv_path} ...")
    fold_map = load_split(args.split_tsv_path)
    from collections import Counter
    counts = Counter(fold_map.values())
    print(f"  Split: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    columns = [
        "identifier", "mzs", "intensities", "smiles", "inchikey",
        "formula", "precursor_formula", "parent_mass", "precursor_mz",
        "adduct", "fold",
    ]

    print(f"Converting MGF (max_peaks={args.max_peaks}) -> {args.out_tsv_path} ...")
    n_written = n_skipped = 0
    with open(args.out_tsv_path, "w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(columns)
        for spec in parse_mgf_gz(args.mgf_gz_path, args.max_peaks):
            title = spec.get("TITLE", "")
            sm = spec.get("SMILES", "")
            fold = fold_map.get(title)

            if not sm or fold is None:
                n_skipped += 1
                continue

            pepmass = spec.get("PEPMASS", "")
            try:
                precursor_mz = float(pepmass)
            except ValueError:
                precursor_mz = ""

            writer.writerow([
                title,
                ",".join(str(m) for m in spec["_mzs"]),
                ",".join(str(v) for v in spec["_intensities"]),
                sm,
                spec.get("INCHIKEY", ""),
                spec.get("FORMULA", ""),
                spec.get("ADDUCT_FORMULA", ""),
                precursor_mz,  # parent_mass proxy — not used at runtime
                precursor_mz,
                spec.get("ADDUCT", ""),
                fold,
            ])
            n_written += 1

    print(f"Done: {n_written} written, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
