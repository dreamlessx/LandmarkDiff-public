# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in LandmarkDiff, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use [GitHub's private vulnerability reporting](https://github.com/dreamlessx/LandmarkDiff-public/security/advisories/new).

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, but we aim for 30 days for critical issues

### Scope

The following are in scope:
- Code execution vulnerabilities in the inference pipeline
- Data leakage from input images
- Model weight tampering
- Dependency vulnerabilities

The following are out of scope:
- Vulnerabilities in upstream dependencies (report to the upstream project)
- Social engineering attacks
- Issues in the Gradio demo when run locally (no auth by default)

## Security Best Practices for Users

- Never expose the Gradio demo to the public internet without authentication
- Rotate any API keys or tokens regularly
- Use the provided Apptainer/Docker containers for isolation
- Keep dependencies updated (`pip install -U landmarkdiff`)
