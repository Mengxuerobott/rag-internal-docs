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