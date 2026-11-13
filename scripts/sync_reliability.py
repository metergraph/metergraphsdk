"""Sync the standalone Core classifier and shared synthetic conformance cases."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("core_checkout", type=Path)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
pairs = [("core/src/metergraph_core/reliability.py", "python/src/metergraph/_reliability.py"),
         ("core/tests/fixtures/reliability.json", "tests/fixtures/reliability.json")]
for source, target in pairs:
    data = (args.core_checkout / source).read_bytes()
    destination = root / target
    if args.check:
        if destination.read_bytes() != data:
            raise SystemExit(f"Core reliability copy differs: {target}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
