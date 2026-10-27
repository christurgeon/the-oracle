# The Oracle

[![CI](https://github.com/christurgeon/the-oracle/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/christurgeon/the-oracle/actions/workflows/ci.yml)

Prediction market trading bot for [Kalshi](https://kalshi.com), a CFTC-regulated exchange. The strategy engine, execution, and analytics are written in Python.

> **Risk notice.** This is a personal research and trading system, published for its
> engineering and its research method. It is **not a profitable strategy**: the research
> notes in `docs/research/` record mostly negative results, and no strategy here has a
> confirmed out-of-sample edge. It can lose real money in `live` mode. Nothing here is
> financial advice. The software is provided "as is", without warranty of any kind (see
> [LICENSE](LICENSE)).

## Project Status

- This is a personal research project with one maintainer.
- There is no support, no roadmap, and no release schedule.
- The maintainer reads issues and pull requests when time allows and can close them without a reply.
- Only the latest `main` is supported.
- You can fork and change the code under the license.
- This project is not affiliated with or endorsed by Kalshi.

## How It Works

The bot runs an async event loop. Each tick it scans Kalshi markets, asks each enabled strategy for trade signals, checks every signal against hard risk limits, and then places an order or logs a shadow trade. A shadow trade is a trade that the bot records but does not place (see the execution modes under [Configuration](#configuration)). Position sizes use fractional Kelly: a fixed fraction of the Kelly-criterion bet size.

```
Orchestrator tick (execution.poll_interval_seconds):
  1. Check kill switch (state/STOP file)
  2. Run circuit breaker checks per strategy
  3. Each enabled Strategy.generate_signals() → list[TradeSignal]
  4. RiskManager validates each signal (exposure, loss, drawdown)
  5. ExecutionEngine places order or logs shadow trade
```

## Strategies

Terms used in this README and in the docs:

- **KXHIGH / KXLOWT / KXRAIN:** Kalshi series for the daily high temperature, the daily low temperature, and daily rain in a city.
- **GFS:** NOAA Global Forecast System. **ECMWF:** European Centre for Medium-Range Weather Forecasts. The bot uses the ensemble forecasts of both.
- **NBM:** National Blend of Models, an NWS forecast product.
- **EMOS:** Ensemble Model Output Statistics, a calibration method that fits a probability distribution to ensemble output.
- **ASOS:** Automated Surface Observing System, the network of weather stations at US airports.
- **NWS CLI:** National Weather Service Daily Climate Report. Kalshi settles the temperature markets on it.
- **mid:** the midpoint of the best bid and the best ask.
- **cell:** a (city, variable) pair, for example `MIA/low`.
- **lead-N:** N days before the observation date.

| Strategy | Markets | Model |
|----------|---------|-------|
| Favorite-Bias | Weather greater/less favorites | Model-free. Isotonic map of market mid to win probability. |
| Weather Edge | KXHIGH / KXLOWT daily temperature brackets | GFS + ECMWF ensemble fraction, then calibration. |
| Intraday EMOS HIGH | Same-day KXHIGH | Shifted EMOS posterior. |
| Intraday ASOS Nowcast | Same-day KXHIGH | Observed ASOS max-so-far plus remaining-gain climatology. |
| KXRAIN shadow observer | Daily-rain markets | Logging only. Not a `Strategy`. Places no orders. |

Whether a strategy trades, runs in shadow, or is parked depends on `enabled` and `shadow_only` under `strategies.*` in [`python/config.yaml`](python/config.yaml). The shipped config runs in shadow mode with Favorite-Bias, the intraday strategies and the KXRAIN observer off, and Weather Edge in shadow only.

An S&P 500 bracket strategy was removed after its backtest showed negative edge.

## Quick Start

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra telegram   # add --extra ml to run the offline fitters
cp .env.example .env                   # then add your Kalshi credentials
uv run python -m python.orchestrator   # runs in shadow mode by default
```

The Kalshi credentials are an API key ID (`ORACLE_KALSHI_API_KEY`) and the path to its RSA private key file (`ORACLE_KALSHI_PRIVATE_KEY_PATH`). Create the key in your Kalshi account's API key settings. You need a Kalshi account and an API key even for shadow mode, because the bot builds its Kalshi client only when credentials are set. Without them the bot starts but has no market data and runs no strategies.

Extras: `dev` (pytest, ruff, mypy), `telegram` (remote control and alerts), `ml` (scikit-learn and others, used only by offline fitting scripts).

Terminal dashboard: `uv run python -m python.dashboard`.

### Configuration

- **Secrets and environment:** `.env`. Variables use the `ORACLE_` prefix. See [`.env.example`](.env.example) for the full list.
- **Strategy, risk, and execution parameters:** [`python/config.yaml`](python/config.yaml).
- **Both are merged** by `python/settings.py`. An environment variable overrides the YAML value.
- **Deployment overlay (optional):** set `ORACLE_CONFIG_OVERLAY` to a YAML file that overrides the `strategies.favorite_bias`, `risk`, `circuit_breakers`, `telegram`, `fills_reconciler` and `kxrain_shadow` sections. It keeps private live values out of `config.yaml`. `uv run python scripts/check_live_config.py --dump` prints the merged config and checks it.

Set `ORACLE_EXECUTION_MODE` (or `execution.mode` in `config.yaml`) to one of:

| Mode | Behavior |
|------|----------|
| `shadow` | Reads live market data, generates signals, logs everything. Places no orders. Default. |
| `paper` | Places orders on the Kalshi demo API. |
| `live` | Places real orders with real capital. |

## Risk Controls

The risk manager enforces these limits on every signal. Limit values are in the `risk:` and `circuit_breakers:` blocks of `python/config.yaml`.

- Per-market exposure cap and total exposure cap, as a share of bankroll.
- Daily loss halt.
- Drawdown from peak: position sizes shrink when the limit is reached.
- Stale-data cutoff: no trading on old market data.
- Correlated-position cap.
- Per-strategy bankroll allocation.
- Per-strategy circuit breakers (rolling P&L, win rate, data freshness).
- Kill switch: create `state/STOP` (or send `/kill` in Telegram) to cancel open orders and halt.

## Repository Layout

```
python/       Bot source, imported as python.*. Module map: python/CLAUDE.md
  config.yaml   Strategy, risk, and execution parameters
scripts/      Offline tools: reports, fitters, data collection. Catalog: docs/scripts-reference.md
tests/        pytest suite
data/         Calibration artifacts. Daily market snapshots are local and gitignored
state/        Live runtime state and logs (trades.jsonl, risk state, kill switch)
deploy/       systemd units and VPS setup scripts
docs/         Runbooks and design notes (below)
```

## Documentation

| Document | Contents |
|----------|----------|
| [`CLAUDE.md`](CLAUDE.md) | Architecture, conventions, and operational constraints. Read before changing runtime code. |
| [`python/CLAUDE.md`](python/CLAUDE.md) | Module map with per-module contracts. |
| [`docs/operations-runbook.md`](docs/operations-runbook.md) | Daily monitoring, recovery, and Telegram commands. |
| [`docs/deploy-guide.md`](docs/deploy-guide.md) | VPS deployment. |
| [`docs/calibration-runbook.md`](docs/calibration-runbook.md) | Refit cadence, calibrator promotion, and rollback. |
| [`docs/scripts-reference.md`](docs/scripts-reference.md) | What each script does. |
| [`docs/tech-debt.md`](docs/tech-debt.md) | Known design items. |
| [`docs/research/`](docs/research/) | Dated research findings behind strategy decisions. |

## Data Collection

`scripts/collect_daily_snapshot.py` collects open markets, ensemble forecasts, and NWS settlements into `data/snapshots/<date>/`. Calibration and backtesting use this data. Snapshots are local, gitignored, and not redistributed because of Kalshi's data terms. Tests that need them skip when they are absent.

## Development

```bash
uv run ruff check python/       # Lint
uv run mypy python/             # Type check
uv run pytest                   # Tests
```

Enable the pre-commit secret scan once per clone: `git config core.hooksPath .githooks` (uses [gitleaks](https://github.com/gitleaks/gitleaks) when installed).

Follow the conventions in [`CLAUDE.md`](CLAUDE.md). The `CLAUDE.md` files are instruction files for Claude Code (Anthropic's coding agent) that double as developer docs.

## Contributing and Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you open an issue or a pull request. It has
the data and secrets rules. Report security problems privately as described in
[SECURITY.md](SECURITY.md).

## License and Data

Code: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

Data sources have their own terms, and the license does not cover them:

- **Kalshi** market data is not included in this repository. Kalshi's developer terms do not allow redistributing API data.
- **Open-Meteo** forecasts are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Weather data by [Open-Meteo.com](https://open-meteo.com/). The free API is for non-commercial use.
- **NWS** climate reports are public domain. ASOS observations come from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/).
