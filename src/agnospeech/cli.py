"""CLI entry point.

    python -m agnospeech.cli conformance --check

The demo scorecard generator was removed in the de-fixing pass. To compute
privatization RESULTS (lexicon-free, both engines) for later comparison, use
``python scripts/compute_results.py`` instead.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agnospeech")
    sub = parser.add_subparsers(dest="cmd", required=True)

    con = sub.add_parser("conformance", help="run the hard-rule conformance suite")
    con.add_argument("--check", action="store_true", help="run all assertions")
    con.add_argument("--allow-cloud-on-raw", action="store_true",
                     help="break a rule live: route raw text to a cloud endpoint")

    args = parser.parse_args(argv)
    if args.cmd == "conformance":
        from .conformance import run as run_conformance

        results = run_conformance(allow_cloud_on_raw=args.allow_cloud_on_raw)
        all_pass = all(a.passed for a in results)
        for a in results:
            tag = "GREEN" if a.passed else "RED"
            print(f"[{tag}] {a.name}\n        {a.detail}")
        print(f"\n{'PASSED' if all_pass else 'FAILED'} · "
              f"{sum(a.passed for a in results)}/{len(results)} green")
        return 0 if all_pass else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
