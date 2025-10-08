#!/usr/bin/env python3
"""Generate benchmark-ready VQA diagnosis CSVs with resolved image paths.

This script reads the modality-specific CSVs under ``datasets/`` and enriches them
with an ``image_path`` column pointing to the corresponding image file on disk.
The resulting CSVs are written to ``datasets/processed`` and keep the original
columns alongside the new path column, preserving the input row order.

Usage (from repository root)::

    uv run python scripts/prepare_vqa_datasets.py

Optional flags allow limiting the generation to a subset of modalities.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"
SOURCE_ROOT = DATASETS_ROOT / "VQA_diagnosis_20250722"
OUTPUT_ROOT = DATASETS_ROOT / "processed"
CSV_VERSION = "20250801"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when expected dataset assets are missing."""


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(f"Expected path does not exist: {path}")
    return path


def _first_existing(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise DatasetNotFoundError(
        "None of the candidate paths exist: "
        + ", ".join(str(path) for path in candidates)
    )


def _write_output(df: pd.DataFrame, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return output_file


# ---------------------------------------------------------------------------
# OCT modality helpers
# ---------------------------------------------------------------------------

_OCTDL_PREFIX_TO_DIR: Mapping[str, str] = {
    "amd": "AMD",
    "dme": "DME",
    "erm": "ERM",
    "no": "NO",
    "rvo": "RVO",
    "vid": "VID",
}

_OCTID_PREFIX_TO_DIR: Mapping[str, str] = {
    "AMRD": "ARMD",
    "CSR": "CSR",
    "DR": "DR",
    "MH": "MH",
    "NORMAL": "NORMAL",
}

_OCTID_PREFIX_TO_EXTS: Mapping[str, Sequence[str]] = {
    "NORMAL": (".jpg", ".jpeg"),
}


def _resolve_oct_sample(row: MutableMapping[str, object], rel_start: Path) -> str:
    dataset = str(row["dataset"])
    imid = str(row["imid"])

    if dataset == "OCTDL":
        prefix = imid.split("_")[0]
        try:
            folder = _OCTDL_PREFIX_TO_DIR[prefix]
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"Unsupported OCTDL prefix '{prefix}' for image id '{imid}'"
            ) from exc

        base_dir = SOURCE_ROOT / "OCTDL" / folder
        filename = f"{imid}.jpg"
        image_path = base_dir / filename

    elif dataset == "OCTID":
        prefix_match = re.match(r"[A-Za-z]+", imid)
        if not prefix_match:
            raise DatasetNotFoundError(f"Unable to parse prefix from OCTID id '{imid}'")
        prefix = prefix_match.group(0)

        try:
            folder = _OCTID_PREFIX_TO_DIR[prefix]
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"Unsupported OCTID prefix '{prefix}' for image id '{imid}'"
            ) from exc

        base_dir = SOURCE_ROOT / "OCTID" / folder
        additional_exts = _OCTID_PREFIX_TO_EXTS.get(prefix, (".jpeg", ".jpg"))
        candidates: List[Path]

        # If the id already includes an extension, respect it first
        suffix = Path(imid).suffix
        if suffix:
            candidates = [base_dir / imid]
        else:
            candidates = [base_dir / f"{imid}{ext}" for ext in additional_exts]

        image_path = _first_existing(candidates)
    else:
        raise DatasetNotFoundError(f"Unknown OCT dataset source '{dataset}'")

    return _relative_to(image_path, rel_start)


# ---------------------------------------------------------------------------
# CFP modality helpers
# ---------------------------------------------------------------------------

_CFP_DATASET_TO_DIR: Mapping[str, str] = {
    "APTOS2019": "APTOS2019",
    "DeepDRiD": "DeepDRiD",
    "HPMI": "HPMI",
    "ODIR-5K": "ODIR-5K",
    "RFMID2.0": "RFMiD2.0",
}

_CFP_DATASET_TO_EXTS: Mapping[str, Sequence[str]] = {
    "APTOS2019": (".png", ".jpg", ".jpeg"),
    "DeepDRiD": (".jpg", ".jpeg"),
    "HPMI": (".jpg", ".jpeg"),
    "ODIR-5K": (".jpg", ".jpeg"),
    "RFMID2.0": (".JPG", ".jpg", ".jpeg"),
}


def _resolve_cfp_sample(row: MutableMapping[str, object], rel_start: Path) -> str:
    dataset_name = str(row["dataset"])
    try:
        folder = _CFP_DATASET_TO_DIR[dataset_name]
    except KeyError as exc:
        raise DatasetNotFoundError(f"Unknown CFP dataset source '{dataset_name}'") from exc

    base_dir = SOURCE_ROOT / folder
    imid = str(row["imid"])
    suffix = Path(imid).suffix

    candidates: List[Path]
    if suffix:
        candidates = [base_dir / imid]
    else:
        extensions = _CFP_DATASET_TO_EXTS.get(dataset_name, (".jpg", ".jpeg"))
        candidates = [base_dir / f"{imid}{ext}" for ext in extensions]

    image_path = _first_existing(candidates)
    return _relative_to(image_path, rel_start)


# ---------------------------------------------------------------------------
# SLO modality helpers
# ---------------------------------------------------------------------------

_SLO_DIAGNOSIS_TO_DIR: Mapping[str, str] = {
    "age-related macular degeneration": "AMD",
    "diabetic retinopathy": "DR",
    "normal": "Healthy",
    "pathologic myopia": "PM",
    "retinal detachment": "RD",
    "retinal vein occlusion": "RVO",
    "uveitis": "Uveitis",
}


def _resolve_slo_sample(row: MutableMapping[str, object], rel_start: Path) -> str:
    diagnosis = str(row["diagnosis"]).strip().lower()
    try:
        folder = _SLO_DIAGNOSIS_TO_DIR[diagnosis]
    except KeyError as exc:
        raise DatasetNotFoundError(f"Unknown SLO diagnosis '{diagnosis}'") from exc

    base_dir = SOURCE_ROOT / "He et al., 2024 (PMID 39567563)" / folder
    imid = str(row["imid"])
    suffix = Path(imid).suffix
    candidates = [base_dir / imid] if suffix else [base_dir / f"{imid}.jpg", base_dir / f"{imid}.jpeg"]
    image_path = _first_existing(candidates)
    return _relative_to(image_path, rel_start)


# ---------------------------------------------------------------------------
# Shared helpers and modality entry points
# ---------------------------------------------------------------------------


def _relative_to(path: Path, rel_start: Path) -> str:
    return Path(os.path.relpath(path, start=rel_start)).as_posix()


def _prepare_modality_csv(
    csv_name: str,
    resolver: Callable[[MutableMapping[str, object], Path], str],
    output_suffix: str,
) -> Path:
    source_csv = DATASETS_ROOT / f"VQA_diagnosis_{CSV_VERSION}_{csv_name}.csv"
    _ensure_exists(source_csv)

    df = pd.read_csv(source_csv)
    rel_start = (OUTPUT_ROOT).resolve()

    df = df.copy()
    df["image_path"] = [resolver(row, rel_start) for row in df.to_dict("records")]

    columns = ["image_path"] + [col for col in df.columns if col != "image_path"]
    df = df[columns]

    output_file = OUTPUT_ROOT / f"VQA_diagnosis_{CSV_VERSION}_{output_suffix}.csv"
    return _write_output(df, output_file)


MODALITY_BUILDERS = {
    "oct": lambda: _prepare_modality_csv("OCT", _resolve_oct_sample, "OCT_with_paths"),
    "cfp": lambda: _prepare_modality_csv("CFP", _resolve_cfp_sample, "CFP_with_paths"),
    "slo": lambda: _prepare_modality_csv("SLO", _resolve_slo_sample, "SLO_with_paths"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modalities",
        nargs="*",
        choices=sorted(MODALITY_BUILDERS.keys()),
        metavar="MOD",
        help="Subset of modalities to regenerate (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = args.modalities or sorted(MODALITY_BUILDERS.keys())

    generated: List[Path] = []
    for modality in selected:
        output = MODALITY_BUILDERS[modality]()
        generated.append(output)

    if generated:
        print("Generated:")
        for path in generated:
            print(f"  - {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except DatasetNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")
