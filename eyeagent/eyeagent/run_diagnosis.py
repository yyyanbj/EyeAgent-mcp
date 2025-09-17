#!/usr/bin/env python3
"""CLI entrypoint: run the full diagnostic workflow once.

Example:
uv run python run_diagnosis.py \
    --patient '{"patient_id":"P001","age":63,"gender":"M"}' \
    --images '[{"image_id":"IMG001","path":"/data/img1.jpg"}]'
"""
import argparse
import json
from .diagnostic_workflow import run_diagnosis


def parse_args():
    ap = argparse.ArgumentParser(description="Run Ophthalmology Diagnostic Workflow")
    ap.add_argument("--patient", required=True, help="Patient JSON string")
    ap.add_argument("--images", required=True, help="Images JSON array; each item has image_id/path")
    return ap.parse_args()


def main():
    args = parse_args()
    patient = json.loads(args.patient)
    images = json.loads(args.images)
    report = run_diagnosis(patient, images)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
