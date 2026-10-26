"""C-ENV-INHERIT: /agent subprocess env must be allowlist-filtered."""

from __future__ import annotations

import os


def test_agent_env_excludes_oracle_secrets(monkeypatch):
    """The subprocess env passed to claude must NOT contain ORACLE_* secrets."""
    from python.notifications import telegram_bot

    # Set fake secrets in the orchestrator's environment
    monkeypatch.setenv("ORACLE_KALSHI_API_KEY", "secret-kalshi-123")
    monkeypatch.setenv("ORACLE_KALSHI_PRIVATE_KEY_PATH", "/tmp/key.pem")
    monkeypatch.setenv("ORACLE_ANTHROPIC_API_KEY", "secret-anthropic-456")
    monkeypatch.setenv("ORACLE_TELEGRAM_BOT_TOKEN", "secret-tgtoken-789")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/opt/oracle")

    # Build the allowlist-filtered env (mirrors the production transformation)
    filtered = {
        k: v for k, v in os.environ.items() if k in telegram_bot._DIAGNOSE_ENV_ALLOWLIST
    }

    # Required: PATH and HOME pass through
    assert filtered.get("PATH") == "/usr/bin"
    assert filtered.get("HOME") == "/opt/oracle"

    # Forbidden: no ORACLE_* secrets
    assert "ORACLE_KALSHI_API_KEY" not in filtered
    assert "ORACLE_KALSHI_PRIVATE_KEY_PATH" not in filtered
    assert "ORACLE_ANTHROPIC_API_KEY" not in filtered
    assert "ORACLE_TELEGRAM_BOT_TOKEN" not in filtered


def test_agent_env_allowlist_has_required_vars():
    """Verify the allowlist tuple contains the documented baseline."""
    from python.notifications import telegram_bot

    required = {
        # POSIX baseline
        "PATH", "HOME", "USER", "LOGNAME", "SHELL",
        # Locale
        "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
        # Claude Code config (HOME-based OAuth resolves /opt/oracle/.claude/.credentials.json)
        "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN",
    }
    actual = set(telegram_bot._DIAGNOSE_ENV_ALLOWLIST)
    assert actual == required, (
        f"Allowlist drifted from spec: "
        f"extra={actual - required}, missing={required - actual}"
    )
