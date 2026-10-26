"""Pre-restart check of the merged live config (config.yaml + overlay + env).

``Settings.load()`` alone does not catch everything that stops the bot:
``validate_strategy_allocations`` only runs after strategies register, so a
bad allocation map passes ``ExecStartPre`` and then crash-loops. This script
runs that check, enforces the reviewed sizing ceilings, and prints two hashes
of the redacted settings dump:

  full  — the whole merged config.
  live  — only the subtrees an overlay may set (``CONFIG_OVERLAY_ROOTS``).

``--expect-hash FILE`` fails when the ``live`` hash differs from the one
pinned in FILE, so a live sizing/enable change cannot reach a restart unless
the pinned hash was updated with it.

Run from the deploy directory as the service user:
  sudo -u oracle .venv/bin/python scripts/check_live_config.py --expect-hash <file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.orchestrator import (  # noqa: E402
    configured_strategy_flags,
    validate_strategy_allocations,
)
from python.settings import CONFIG_OVERLAY_ROOTS, Settings, config_overlay_path  # noqa: E402

_SECRET_FIELDS = frozenset(
    {
        "kalshi_api_key",
        "kalshi_private_key_path",
        "kalshi_private_key_password",
        "telegram_bot_token",
        "telegram_owner_chat_id",
        "healthcheck_ping_url",
    }
)


def sizing_violations(settings: Settings) -> list[str]:
    """Sizing knobs past the reviewed ceilings. Raising one must be deliberate."""
    fb = settings.strategies.favorite_bias
    risk = settings.risk
    checks = (
        (fb.kelly_fraction <= 0.60, f"favorite_bias.kelly_fraction={fb.kelly_fraction} > 0.60"),
        (
            fb.max_contracts_per_trade <= 100,
            f"favorite_bias.max_contracts_per_trade={fb.max_contracts_per_trade} > 100",
        ),
        (
            risk.max_single_market_pct <= 0.10,
            f"risk.max_single_market_pct={risk.max_single_market_pct} > 0.10",
        ),
        (
            risk.max_daily_loss_pct <= 0.15,
            f"risk.max_daily_loss_pct={risk.max_daily_loss_pct} > 0.15",
        ),
        (
            risk.max_daily_loss_pct < risk.max_drawdown_pct,
            "risk.max_daily_loss_pct must stay below risk.max_drawdown_pct",
        ),
        (
            risk.drawdown_size_reduction >= 0.5,
            f"risk.drawdown_size_reduction={risk.drawdown_size_reduction} < 0.50",
        ),
    )
    return [msg for ok, msg in checks if not ok]


def redacted_dump(settings: Settings) -> dict[str, Any]:
    return settings.model_dump(mode="json", exclude=set(_SECRET_FIELDS))


def live_subset(dump: dict[str, Any]) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for root in CONFIG_OVERLAY_ROOTS:
        node: Any = dump
        for key in root:
            node = node[key]
        subset[".".join(root)] = node
    return subset


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--expect-hash", type=Path, help="file holding the pinned live hash")
    parser.add_argument("--dump", action="store_true", help="print the redacted settings dump")
    args = parser.parse_args(argv)

    settings = Settings.load()
    print(f"overlay: {config_overlay_path() or 'none'}")

    validate_strategy_allocations(
        settings.risk.strategy_allocations, configured_strategy_flags(settings)
    )
    print("allocations: ok")

    violations = sizing_violations(settings)
    for msg in violations:
        print(f"sizing ceiling exceeded: {msg}", file=sys.stderr)
    if violations:
        return 1
    print("sizing ceilings: ok")

    dump = redacted_dump(settings)
    if args.dump:
        print(json.dumps(dump, indent=2, sort_keys=True))
    live_hash = digest(live_subset(dump))
    print(f"full {digest(dump)}")
    print(f"live {live_hash}")

    if args.expect_hash is not None:
        expected = args.expect_hash.read_text().split()[0]
        if expected != live_hash:
            print(
                f"live config hash {live_hash} != pinned {expected} ({args.expect_hash}). "
                "Update the pinned hash in the same change as the overlay edit.",
                file=sys.stderr,
            )
            return 1
        print("live hash: matches pin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
