#!/usr/bin/env python3
"""
Creates the repo folder structure:
- keeps existing Compoent_Images/ and Compoent_Images.zip if present
- creates src/, data/batches/, outputs/
- creates placeholder batch folder data/batches/batch_001/
- optionally creates README.md and requirements.txt (if missing)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ensure_file(p: Path, content: str) -> None:
    if not p.exists():
        p.write_text(content, encoding="utf-8")

def main() -> None:
    # Must keep these (do not create if absent, just report)
    comp_dir = ROOT / "Compoent_Images"
    comp_zip = ROOT / "Compoent_Images.zip"

    print(f"Project root: {ROOT}")

    if comp_dir.exists():
        print(f"✅ Found: {comp_dir}")
    else:
        print(f"⚠️ Not found (kept if you have it): {comp_dir}")

    if comp_zip.exists():
        print(f"✅ Found: {comp_zip}")
    else:
        print(f"⚠️ Not found (kept if you have it): {comp_zip}")

    # Create structure
    ensure_dir(ROOT / "src")
    ensure_dir(ROOT / "data" / "batches" / "batch_001")
    ensure_dir(ROOT / "outputs")

    # Minimal README + requirements if missing
    ensure_file(
        ROOT / "requirements.txt",
        "opencv-python\nnumpy\n"
    )

    ensure_file(
        ROOT / "README.md",
        "# CV_Project\n\n"
        "Batch dataset builder scripts (Edges baseline + GrabCut improved).\n\n"
        "## Setup\n"
        "```bash\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "```\n\n"
        "## Batches\n"
        "Put new images into:\n"
        "`data/batches/<batch_name>/`\n\n"
        "## Run\n"
        "Edges baseline:\n"
        "```bash\n"
        "python src/build_batch_edges.py --in data/batches/batch_001 --class brake_caliper\n"
        "```\n\n"
        "GrabCut improved:\n"
        "```bash\n"
        "python src/build_batch_grabcut.py --in data/batches/batch_001 --class brake_caliper\n"
        "```\n"
    )

    print("\n✅ Structure ready:")
    print(" - src/")
    print(" - data/batches/batch_001/")
    print(" - outputs/")
    print(" - requirements.txt (if missing, created)")
    print(" - README.md (if missing, created)")

if __name__ == "__main__":
    main()
