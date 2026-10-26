# Security Policy

## Report a Vulnerability

Do not open a public issue for a security problem.

Use GitHub private vulnerability reporting. Open the **Security** tab and click
**Report a vulnerability**, or use this link:
https://github.com/christurgeon/the-oracle/security/advisories/new

## Scope

In scope:

- Code that signs Kalshi API requests or reads the RSA private key.
- Code that can leak credentials, tokens, or account data to logs, files, or alerts.
- The owner-only check on Telegram commands.
- Secrets in this repository, its git history, its issues, or its pull requests.
- Weaknesses in the CI workflow.

Out of scope:

- Kalshi's exchange, website, or API. Report those to Kalshi.
- Trading losses, strategy quality, or model errors. These are not vulnerabilities.
- A dependency vulnerability that this code cannot reach.
- Your own deployment: your server, your keys, or your configuration.

## What to Include

- The affected file and commit.
- Steps to reproduce. Use made-up credentials.
- The impact.

Do not include real keys, tokens, `.env` contents, or account details — yours or anyone
else's. Do not include Kalshi market data.

## If Your Own Credentials Leaked

Revoke them first.

- Delete the Kalshi API key in your Kalshi account settings. Create a new key.
- Revoke a Telegram bot token with @BotFather.

## Response

This is a personal project with one maintainer. There is no response-time commitment and
no bug bounty. The maintainer reads reports when time allows. Fixes go to `main` only.
There are no releases, so only the latest `main` is supported.
