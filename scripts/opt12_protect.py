"""OPT-12: re-hash the protected artifacts (movd.opt12.PROTECTED); exit 1 on any mismatch.

    python scripts/opt12_protect.py --tag start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import opt12  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = {}
    for name, (path, expected) in opt12.PROTECTED.items():
        got = opt12.sha256(path) if Path(path).exists() else None
        rows[name] = {"path": Path(path).as_posix(), "expected": expected, "actual": got, "ok": got == expected}
    ok = all(r["ok"] for r in rows.values())
    opt12.save_json(opt12.EVID / f"protected_{a.tag}.json", {"at_kyiv": opt12.stamp(), "all_ok": ok, "artifacts": rows})
    print({k: v["ok"] for k, v in rows.items()}, "ALL OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
