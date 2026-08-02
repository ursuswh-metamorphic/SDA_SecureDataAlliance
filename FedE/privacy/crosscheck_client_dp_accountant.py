"""Cross-check the client-level RDP accountant against Opacus.

This check uses the experiment's actual full-client participation schedule:
q=1, replace-one sensitivity handled by the mechanism's noise scale, and one
accountant step per released aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opacus.accountants.analysis import rdp as opacus_rdp

from rdp_accountant import DEFAULT_ORDERS, compute_epsilon, find_noise_multiplier


def opacus_epsilon(
    *, steps: int, sigma: float, sample_rate: float, delta: float
) -> tuple[float, float]:
    orders = [order for order in DEFAULT_ORDERS if order != float("inf")]
    rdp_values = opacus_rdp.compute_rdp(
        q=sample_rate,
        noise_multiplier=sigma,
        steps=steps,
        orders=orders,
    )
    epsilon, best_alpha = opacus_rdp.get_privacy_spent(
        orders=orders,
        rdp=rdp_values,
        delta=delta,
    )
    return float(epsilon), float(best_alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 15])
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0, 20.0]
    )
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    passed = True
    for steps in args.steps:
        for target_epsilon in args.epsilons:
            sigma = find_noise_multiplier(
                target_epsilon=target_epsilon,
                num_steps=steps,
                sample_rate=args.sample_rate,
                delta=args.delta,
            )
            native_epsilon, native_alpha = compute_epsilon(
                steps, sigma, args.sample_rate, args.delta
            )
            independent_epsilon, independent_alpha = opacus_epsilon(
                steps=steps,
                sigma=sigma,
                sample_rate=args.sample_rate,
                delta=args.delta,
            )
            absolute_difference = abs(native_epsilon - independent_epsilon)
            row_passed = absolute_difference <= args.tolerance
            passed &= row_passed
            rows.append(
                {
                    "steps": steps,
                    "sample_rate": args.sample_rate,
                    "delta": args.delta,
                    "target_epsilon": target_epsilon,
                    "noise_multiplier": sigma,
                    "native_epsilon": native_epsilon,
                    "native_best_alpha": native_alpha,
                    "opacus_epsilon": independent_epsilon,
                    "opacus_best_alpha": independent_alpha,
                    "absolute_difference": absolute_difference,
                    "pass": row_passed,
                }
            )

    report = {
        "privacy_unit": "one complete federated client",
        "adjacency": "replace-one",
        "accounting_schedule": "one Gaussian mechanism per released round",
        "crosscheck": "native RDP accountant vs Opacus 1.6",
        "absolute_tolerance": args.tolerance,
        "pass": passed,
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not passed:
        raise SystemExit("accountant cross-check failed")


if __name__ == "__main__":
    main()
