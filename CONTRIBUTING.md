# Contributing

The Oracle is a personal research project with one maintainer. Read
[Project Status](README.md#project-status) first.

There is no support. The maintainer reviews pull requests when time allows. A pull request
can stay open, or be closed, without a review.

## What to Send

- Bug fixes with a test that fails before the fix.
- Fixes to tests, type errors, or documentation.

Large changes and new strategies are unlikely to merge. Fork the project for those. You do
not need permission to fork.

## Data Rules

Kalshi's Developer Agreement does not allow redistribution of Kalshi API data. These rules
apply to code, tests, issues, pull requests, and comments.

- Do not add Kalshi market data. This includes market lists, prices, orderbooks, trades,
  fills, settlements, and screenshots of them.
- Use made-up values in test fixtures. Do not copy fixtures from real API responses.
- Do not commit raw downloads from any data source. `data/snapshots/` is gitignored for this
  reason.
- If you add a fitted artifact under `data/`, say how you made it in the pull request. It
  must not contain raw rows.

## Secrets

- Never paste `.env` contents, API key IDs, private keys, Telegram bot tokens, or Telegram
  chat IDs.
- Never paste account details: balances, positions, order IDs, or fills.
- Enable the pre-commit secret scan once per clone: `git config core.hooksPath .githooks`.
  Install [gitleaks](https://github.com/gitleaks/gitleaks) so the hook can run.
- If you expose a credential, revoke it first. Then tell the maintainer as described in
  [SECURITY.md](SECURITY.md) so the text can be removed.

## Checks

CI runs these on every pull request. Run them before you push.

```bash
uv sync --frozen --extra dev --extra telegram --extra ml
uv run ruff check python/
uv run mypy python/
uv run pytest
```

CI also runs gitleaks over the full git history.

- Do not run `ruff format`. The code base does not use it, and reformatting makes the diff
  hard to review.
- `mypy` runs in strict mode on `python/`. New code must pass it.

## Code Conventions

Follow the "Coding Conventions" section in [CLAUDE.md](CLAUDE.md). Before you change runtime
code, read the module contracts in [python/CLAUDE.md](python/CLAUDE.md).

Keep the safe defaults. A change must not make the bot place orders by default. `shadow`
stays the default execution mode.

## Commits and Merges

- Keep each pull request to one change.
- The maintainer merges by rebase or fast-forward. Your commits land on `main` with your
  author name and email.
- Use a commit email that you are willing to make public. Your GitHub noreply address works.
- Contributions are licensed under Apache-2.0, as section 5 of the [LICENSE](LICENSE) states.
  There is no CLA.

## Conduct

Be civil. The maintainer removes abusive content and can block accounts.
