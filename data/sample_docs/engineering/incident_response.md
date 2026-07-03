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