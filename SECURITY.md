# Security policy

FedDisclosure Lab is a research benchmark. It is not a production privacy or security product, and nothing here is certified. Please do not deploy it to protect real data.

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please report privately and do not open a public issue.

- Use GitHub's **Report a vulnerability** button on the Security tab of this repository (private vulnerability reporting).
- Include the affected file or component, steps to reproduce, and the impact you see.

You can expect an acknowledgement within 7 days. Fixes and disclosure timing are agreed with the reporter.

## In scope

- Bugs in the analytics gateway, query-history policy or budget ledger that bypass a stated control.
- Secrets, tokens or personal data committed by mistake.
- Unsafe defaults in the Docker or Compose files.

## Out of scope

- The privacy leakage the project measures on purpose, such as update inversion or membership inference results. Those are findings, not vulnerabilities. Open an issue or a pull request instead.
- Attacks that need a deployment this project never claims to support.
