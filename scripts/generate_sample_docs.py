"""
scripts/generate_sample_docs.py
────────────────────────────────
Generates a set of realistic fake internal company documents so you can
test the full pipeline without needing real confidential files.

Creates:
  data/sample_docs/hr/employee_handbook.md
  data/sample_docs/hr/leave_policy.md
  data/sample_docs/engineering/onboarding_guide.md
  data/sample_docs/engineering/incident_response.md
  data/sample_docs/finance/expense_policy.md
  data/sample_docs/legal/data_retention_policy.md
  data/sample_docs/general/it_security_policy.md

Run:
    python scripts/generate_sample_docs.py
"""

import os
from pathlib import Path

BASE_DIR = Path("data/sample_docs")

DOCS = {
    # ── HR ────────────────────────────────────────────────────────────────────
    "hr/employee_handbook.md": """
# Employee Handbook — Acme Corp

## Welcome
Welcome to Acme Corp. This handbook outlines the policies, procedures, and
expectations that govern our working environment. All employees are expected
to read and adhere to its contents.

## Employment Types
- **Full-time**: 40 hours per week. Eligible for all benefits from day one.
- **Part-time**: Under 30 hours per week. Eligible for prorated benefits.
- **Contractor**: Independent contractors are not eligible for company benefits.

## Probationary Period
All new employees serve a 90-day probationary period. During this time:
- Performance is reviewed at day 30, 60, and 90.
- Either party may terminate employment with one week's notice.
- Full benefits begin on the first day regardless of probationary status.

## Code of Conduct
Employees must:
1. Treat all colleagues, clients, and partners with respect.
2. Avoid conflicts of interest and disclose any potential ones to HR.
3. Protect confidential company information.
4. Comply with all applicable laws and regulations.

Violations of the code of conduct may result in disciplinary action up to
and including termination.

## Dress Code
Acme Corp maintains a smart-casual dress code in the office. On client-facing
days, business casual is expected. Remote employees are expected to dress
appropriately for video calls.

## Performance Reviews
Formal performance reviews are conducted annually in December.
Mid-year check-ins are held in June. Managers may schedule additional
reviews at any time. Performance ratings influence annual compensation adjustments.

## Termination
Voluntary resignation requires two weeks written notice. Involuntary termination
may be immediate in cases of gross misconduct. All company equipment, access
credentials, and confidential materials must be returned on the last working day.
""",

    "hr/leave_policy.md": """
# Leave Policy — Acme Corp

## Vacation Leave
Full-time employees accrue 15 days of paid vacation per year (1.25 days/month).
After 3 years of service, accrual increases to 20 days per year.
After 5 years, it increases to 25 days per year.

Unused vacation may be carried over up to a maximum of 10 days into the next
calendar year. Days above 10 are forfeited on January 1st.

Vacation requests must be submitted at least 14 calendar days in advance via
the HR portal (hr.acme.internal). Manager approval is required. Requests
during blackout periods (last two weeks of each quarter) require VP-level approval.

## Sick Leave
Employees receive 10 days of paid sick leave per year. Sick leave does not
carry over. A doctor's note is required for absences exceeding three consecutive days.

## Parental Leave
Primary caregivers (birth or adoption) receive 16 weeks of fully paid leave.
Secondary caregivers receive 4 weeks of fully paid leave.
Leave must commence within 12 months of the birth or adoption date.
Parental leave is available after 6 months of employment.

## Bereavement Leave
Up to 5 days of paid leave for the loss of an immediate family member
(spouse, child, parent, sibling). Up to 3 days for extended family.
Additional unpaid leave may be arranged with manager approval.

## Jury Duty
Employees called for jury duty receive full pay for up to 10 business days.
A copy of the summons must be provided to HR. Employees are expected to return
to work on days when court is not in session.

## Public Holidays
Acme Corp observes 11 federal public holidays per year. A full list is
published in the HR portal at the start of each calendar year.

## Unpaid Leave of Absence
Extended unpaid leave (up to 12 weeks) may be granted for personal or medical
reasons under FMLA. Requests must be submitted in writing 30 days in advance
where possible. Health benefits continue during approved FMLA leave.
""",

    # ── Engineering ───────────────────────────────────────────────────────────
    "engineering/onboarding_guide.md": """
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
""",

    "engineering/incident_response.md": """
# Incident Response Runbook

## Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|---------|
| SEV-1 | Complete outage or data loss | 15 minutes | API down, DB corruption |
| SEV-2 | Major feature broken, >20% users affected | 30 minutes | Checkout failure, auth errors |
| SEV-3 | Partial degradation, workaround exists | 4 hours | Slow queries, minor UI bug |
| SEV-4 | Cosmetic or low-impact | Next business day | Typo, minor display issue |

## Response Process

### Step 1: Declare the Incident
- For SEV-1/SEV-2: Post in `#incidents` on Slack with:
  `@incident SEV-1: [brief description]`
  This auto-creates an incident channel and pages the on-call engineer.
- For SEV-3/4: Create a Linear ticket with the Incident label.

### Step 2: Assign Roles
- **Incident Commander (IC)**: Coordinates response; first on-call engineer by default.
- **Tech Lead**: Owns technical investigation and fix.
- **Comms Lead**: Updates internal stakeholders and, for SEV-1, the status page.

### Step 3: Investigate
1. Check Datadog dashboards for anomaly timeline.
2. Search Kibana for error spikes around the start time.
3. Review recent deployments in GitHub Actions.
4. Check cloud provider status pages (AWS, GCP).

### Step 4: Mitigate
Common first actions:
- Rollback the last deployment: `make rollback env=prod service=<name>`
- Scale up the affected service: adjust in Kubernetes HPA config.
- Enable feature flag to disable the affected feature.
- Failover to the secondary region if primary is unavailable.

### Step 5: Resolve & Post-Mortem
- Mark incident resolved in PagerDuty.
- Post resolution update in the incident channel.
- For SEV-1/SEV-2: a written post-mortem is due within 5 business days.
  Use the post-mortem template in Notion: `Engineering > Post-Mortems > Template`.

## On-Call Rotation
On-call engineers rotate weekly (Monday 9am PT handoff).
The schedule is in PagerDuty. All engineers with > 6 months tenure participate.
On-call engineers receive a $200/week stipend for their week on rotation.
Pager-free hours are 10pm–7am local time except for SEV-1.
""",

    # ── Finance ───────────────────────────────────────────────────────────────
    "finance/expense_policy.md": """
# Expense Reimbursement Policy

## Overview
Acme Corp reimburses employees for reasonable and necessary business expenses
incurred while performing job duties. Personal expenses will not be reimbursed.

## Submission Process
1. Submit expenses via **Concur** (concur.acme.internal) within **30 calendar days**
   of the expense date. Late submissions require VP Finance approval.
2. Attach itemised receipts for all expenses **over $25**.
3. Select the correct cost centre and add a brief business justification.
4. Approved reimbursements are paid in the next regular payroll cycle.

## Approval Thresholds
| Amount | Approver |
|--------|----------|
| Up to $500 | Direct manager |
| $500–$2,500 | Director-level |
| $2,500–$10,000 | VP-level |
| Over $10,000 | CFO |

## Travel

### Flights
- Book at least 14 days in advance where possible.
- Economy class for flights under 6 hours. Business class for 6+ hours with VP approval.
- Use the preferred booking tool (Navan) to qualify for corporate rates.

### Hotels
- Up to $250/night in Tier 1 cities (NYC, SF, London, Tokyo).
- Up to $175/night in other cities.
- Book through Navan for corporate rates. Personal loyalty points are acceptable.

### Per Diem (Meals)
- Breakfast: $20 | Lunch: $30 | Dinner: $60
- Alcohol is not reimbursable unless explicitly approved for a client event.

### Ground Transport
- Uber/Lyft: reimbursable for business travel. Select UberX or Lyft Standard.
- Personal car mileage: reimbursed at the current IRS rate ($0.67/mile in 2026).
- Parking at company offices is not reimbursed (use the commuter benefit programme).

## Software & Subscriptions
Individual SaaS subscriptions under $50/month may be expensed with manager approval.
Subscriptions over $50/month require IT + Finance approval before purchase.

## Non-Reimbursable Expenses
- Personal meals not related to a business meeting
- Gym memberships or personal wellness (covered by the $100/month wellness stipend)
- Traffic/parking fines
- First-class upgrades (unless booked for a medical reason with HR approval)
- Entertainment for family members
""",

    # ── Legal ─────────────────────────────────────────────────────────────────
    "legal/data_retention_policy.md": """
# Data Retention and Deletion Policy

## Purpose
This policy establishes minimum and maximum retention periods for data held
by Acme Corp, in compliance with GDPR, CCPA, and applicable US federal law.

## Retention Schedule

| Data Category | Minimum Retention | Maximum Retention | Legal Basis |
|--------------|------------------|------------------|-------------|
| Employee records | Duration of employment | 7 years post-departure | Tax / labour law |
| Payroll records | 3 years | 7 years | FLSA |
| Customer contracts | Duration + 5 years | Duration + 7 years | Contract law |
| Financial records | 7 years | 10 years | SOX, IRS |
| Email correspondence | 1 year | 7 years | Business need |
| Application logs | 90 days | 1 year | Security audit |
| Customer PII | Duration of relationship | 3 years post-churn | GDPR / CCPA |
| Security audit logs | 1 year | 3 years | SOC 2 |
| Marketing data | Until opt-out | 3 years | GDPR consent |

## Data Deletion Process
At the end of the maximum retention period, data must be:
1. Securely deleted from all primary storage systems.
2. Purged from all backup systems within the next scheduled backup rotation.
3. Documented in the Data Deletion Log maintained by the Security team.

Deletion requests from data subjects (GDPR Article 17 / CCPA) must be processed
within **30 calendar days**. Legal holds override deletion schedules — consult
Legal before deleting any data subject to an active hold.

## Cloud and SaaS Data
Data stored in third-party SaaS tools (Salesforce, Workday, Slack, etc.) is
subject to the same retention schedules. System owners are responsible for
configuring retention settings in each tool. Annual audits are conducted by
the Security & Compliance team.

## Breach and Incident Data
Data related to a confirmed security incident must be preserved for a minimum
of 3 years from the date of discovery, regardless of the standard schedule.
Do not delete or alter any data if you suspect it may be subject to litigation.

## Responsibility
- **Data owners**: Business unit leaders who own the data-generating process.
- **Security team**: Enforce technical controls and audit compliance.
- **Legal**: Advise on legal holds and regulatory requirements.
- **IT**: Implement deletion scripts and manage backup systems.

Questions? Contact the Data Protection Officer at dpo@acme.com.
""",

    # ── IT Security ───────────────────────────────────────────────────────────
    "general/it_security_policy.md": """
# IT Security Policy

## Password Requirements
All employee passwords must meet the following standards:
- Minimum **12 characters** in length.
- Must contain: uppercase letter, lowercase letter, number, and special character.
- Must not contain your name, email address, or common dictionary words.
- Passwords must be rotated every **90 days**.
- Previous 10 passwords may not be reused.
- Multi-factor authentication (MFA) is **mandatory** for all company accounts.

Use the company-approved password manager (1Password) — do not write passwords down
or store them in plain text files.

## Device Security
- All laptops must have full-disk encryption enabled (FileVault on macOS, BitLocker on Windows).
- Screen lock must activate after **5 minutes** of inactivity.
- Lost or stolen devices must be reported to IT (it-security@acme.com) within **1 hour**.
- IT will remotely wipe lost devices.
- Personal devices may not be used for company work unless enrolled in MDM (Jamf).

## Network Security
- Company VPN (GlobalProtect) must be active when accessing internal services.
- Public Wi-Fi networks must not be used without VPN.
- Do not connect untrusted USB drives or external storage to company devices.

## Acceptable Use
Company devices and accounts are for business use. Incidental personal use is
permitted but subject to monitoring. The following are strictly prohibited:
- Installing unlicensed software.
- Accessing or distributing content that violates copyright law.
- Using company resources for personal commercial activity.
- Attempting to access systems or data you are not authorised for.

## Phishing and Social Engineering
- Do not click links or open attachments from unexpected or suspicious emails.
- Report suspected phishing to security@acme.com or via the Outlook "Report Phishing" button.
- IT will never ask for your password via email, Slack, or phone.

## Data Handling
- **Confidential data** (PII, financial, IP) must not be shared via personal email or
  consumer cloud storage (Google Drive personal, Dropbox free tier, etc.).
- Use company-approved tools: Google Workspace Drive, SharePoint, or Notion.
- Encrypt files containing PII before sending via email.

## Security Training
All employees must complete annual security awareness training via Workday.
Engineers must additionally complete the OWASP Top 10 training module.
Failure to complete training within the deadline may result in access suspension.

## Reporting Security Incidents
Report any suspected security incidents immediately:
- Slack: `#security-incidents` (monitored 24/7)
- Email: security@acme.com
- After hours emergency: call the security hotline at +1-800-XXX-XXXX

Do not attempt to investigate or remediate a security incident yourself.
Preserve all evidence and await instructions from the Security team.
""",
}


def generate_docs() -> None:
    """Write all sample documents to disk."""
    created = 0
    for rel_path, content in DOCS.items():
        full_path = BASE_DIR / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip(), encoding="utf-8")
        print(f"  ✓ {full_path}")
        created += 1

    print(f"\nCreated {created} sample documents in '{BASE_DIR}'")
    print("Run ingestion next:\n  python -m ingestion.embedder")


if __name__ == "__main__":
    generate_docs()
