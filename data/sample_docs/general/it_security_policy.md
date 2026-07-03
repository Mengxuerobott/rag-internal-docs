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