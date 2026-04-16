import argparse
import csv
import pathlib


def scale_file(path: pathlib.Path, percent_drop: float, inplace: bool) -> pathlib.Path:
    scale = max(0.0, 1.0 - (percent_drop / 100.0))
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        fieldnames = reader.fieldnames
        if len(fieldnames) < 2:
            raise ValueError(f"Expected at least 2 columns in {path}, got {len(fieldnames)}")
        r0_col = fieldnames[1]
        rows = []
        for row in reader:
            raw = row.get(r0_col, "")
            if raw != "":
                try:
                    value = float(raw)
                    row[r0_col] = f"{value * scale:.12g}"
                except ValueError:
                    pass
            rows.append(row)

    if inplace:
        out_path = path
    else:
        out_path = path.with_name(f"{path.stem}_r0_scaled_{percent_drop:g}pct{path.suffix}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale R0 column in specific LUT CSV files by a percent drop.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more R0-SoC-T*.csv files to scale",
    )
    parser.add_argument("--percent-drop", type=float, required=True, help="Percent drop for R0 column (e.g., 20)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite files in place")
    args = parser.parse_args()

    for file_path in args.files:
        path = pathlib.Path(file_path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        out_path = scale_file(path, args.percent_drop, args.inplace)
        print(f"{path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
