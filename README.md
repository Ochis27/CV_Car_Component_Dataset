# CV_Project

Batch dataset builder scripts (Edges baseline + GrabCut improved).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Batches
Put new images into:
`data/batches/<batch_name>/`

## Run
Edges baseline:
```bash
python src/build_batch_edges.py --in data/batches/batch_001 --class brake_caliper
```

GrabCut improved:
```bash
python src/build_batch_grabcut.py --in data/batches/batch_001 --class brake_caliper
```
