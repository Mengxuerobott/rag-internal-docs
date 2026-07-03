# Engineering Onboarding Guide

## Week 1: Environment Setup

### Day 1
1. IT will provision your laptop (MacBook Pro M3 or Linux workstation by request).
2. Log in to Okta (SSO) with your @acme.com email.
3. Join the `#engineering` and `#new-joiners` Slack channels.
4. Request access to GitHub org `acme-corp` via the #it-access Slack channel.
5. Install the company VPN (GlobalProtect). VPN is required for all internal services.

### Day 2–3
6. Clone the main monorepo: `git clone git@github.com:acme-corp/platform.git`
7. Follow `platform/README.md` to set up your local dev environment.
   The setup script handles Homebrew, Docker Desktop, Python 3.12, and Node 22.
8. Run the test suite: `make test`. All tests must be green before you start coding.
9. Book 30-min 1:1s with each member of your team (calendar invites in the
   onboarding Notion page).

### Day 4–5
10. Complete mandatory security training in Workday (takes ~2 hours).
11. Read the Architecture Decision Records (ADRs) in `platform/docs/adr/`.
12. Shadow a production deployment with your assigned onboarding buddy.

## First 30 Days
- Complete at least one small "good first issue" ticket (labelled in Linear).
- Attend engineering all-hands (first Wednesday of each month, 2pm PT).
- Set up your development environment for the team's primary service.

## Development Workflow
1. All work happens on feature branches: `feat/<ticket-id>-short-description`.
2. Open a pull request against `main` — require at least 2 approvals.
3. All PRs must pass CI (GitHub Actions): lint, type-check, unit tests, integration tests.
4. Deployments to staging happen automatically on merge to `main`.
5. Production deployments are gated behind a manual approval in GitHub Actions.

## Key Internal Services
| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | grafana.acme.internal | Metrics & dashboards |
| Kibana | kibana.acme.internal | Log search |
| Vault | vault.acme.internal | Secrets management |
| Linear | linear.acme.internal | Issue tracker |
| Notion | notion.acme.internal | Docs & wikis |
| Datadog | datadog.acme.internal | APM & alerting |

## Getting Help
- Slack: `#engineering-help` for general questions.
- On-call rotation: check PagerDuty. Don't be afraid to page if something is breaking.
- Your onboarding buddy is your first point of contact for the first 90 days.