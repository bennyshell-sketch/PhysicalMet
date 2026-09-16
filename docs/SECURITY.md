# Security and privacy

Written for someone deciding whether Dive is safe to deploy against real
patient data. It states what is implemented, what is not, and what is
deliberately impossible.

## The one claim not made

**Dive is not unhackable.** No software is. Anything that claims otherwise is
either misunderstanding the problem or selling something. What follows raises
the cost of an attack and makes a successful one visible afterwards, which is
the achievable goal.

## The privacy question, answered honestly

If Dive sends text to Claude or Gemini, that provider processes it. There is no
header, setting, tier or contract that makes a provider unable to see what you
send them. Anyone who tells you otherwise is describing a *retention* policy,
not a *processing* one, and those are different things.

So Dive does not try to hide data from providers. It controls what reaches them.

### The egress boundary

Every outbound model call, from every module, passes one `EgressGuard`. There is
no second path, which is what makes the audit log a complete record rather than
a sample.

| Mode | What may leave |
|---|---|
| `none` | Nothing participant-derived. |
| `deidentified` *(default)* | Clinical values only. No identifiers, no dates, no free text. Age in 5-year bands with a `90+` cap. Values rounded. |
| `full` | Refused by the guard. |

`full` exists so the configuration surface is explicit rather than implied by
absence. The guard raises rather than honouring it.

### Fail-closed, everywhere

- A field not on the allowlist is **dropped**, not passed through on the
  assumption it looks harmless.
- A payload still matching an identifier pattern *after* de-identification is
  **blocked**, not redacted and sent. Silently rewriting it would conceal that
  something upstream put an identifier there.
- If the scanner itself raises, the egress is **blocked**.
- An unlisted destination is **blocked**, and each destination is reviewed
  separately — clearing a payload for one provider does not clear it for
  another.

### What is honestly not claimed

De-identified is not anonymous. A sufficiently unusual combination of
measurements, cross-referenced against an external dataset, is a real if remote
re-identification route. Two configurations reduce that to zero:

```yaml
privacy:
  egress_mode: none      # no profile egress at all
  offline_only: true     # no external call of any kind
```

or point `DIVE_LOCAL_LLM_URL` at a model on your own hardware. The guard still
runs in that case, in case a "local" URL is misconfigured to a public one.

## Implemented controls

| Control | Implementation | Test |
|---|---|---|
| Tamper-evident audit | Hash-chained JSONL; each entry carries the previous entry's SHA-256 | Deleting or editing a line is detected |
| Audit redaction | Identifiers stripped before the record is written | Stored entries contain no identifier |
| Encryption at rest | AES-GCM, key from `DIVE_ENCRYPTION_KEY` | Tampered ciphertext refuses to decrypt |
| Password storage | scrypt, per-password salt, 12-char minimum | Same password hashes differently each time |
| Account lockout | Configurable threshold and duration | Lockout engages and reports uniformly |
| Session tokens | HMAC-signed, expiring, constant-time compare | A client-edited role fails verification |
| Username enumeration | Identical error and timing for unknown user and wrong password | Messages assert equal |
| Upload validation | Size cap, extension allowlist, magic bytes, path traversal, null bytes | Each rejection asserted |
| Formula injection | Leading `=`, `+`, `-`, `@` neutralised on export | Each prefix asserted |
| Secret handling | Held by reference; `repr()` shows a fingerprint | Secret absent from `repr`, `str`, fingerprint |

Roles: `viewer` (aggregates), `analyst` (individual records, models, ingestion),
`admin` (users, model promotion, privacy policy, export).

Run `python -m pytest tests/test_security.py tests/test_privacy.py -q` to
exercise all of it.

## Not covered here — the deployer's responsibility

These cannot be solved in application code and must be in place before real
patient data is loaded:

1. TLS termination, certificate management, and HSTS.
2. Network isolation; Dive should not be directly internet-facing.
3. OS and dependency patching, with a documented cadence.
4. Key custody and rotation. `DIVE_ENCRYPTION_KEY` belongs in a secret manager,
   never in the repository, an image, or an environment file in version control.
5. Encrypted backups **with tested restores**. An untested backup is not a backup.
6. Shipping the audit log to append-only storage on a separate host. A
   hash chain detects edits; it does not survive an attacker who can rewrite
   the whole file.
7. An independent penetration test and a written threat model.
8. Rate limiting and DoS protection at the reverse proxy.

## Governance — settle these before selling

**Data protection.** UK GDPR and DPA 2018 apply. A DPIA is required, and it must
cover the model providers specifically as processors: which fields reach them,
under what lawful basis, under what retention terms, and in which jurisdiction.

**Medical device regulation.** This is the one that changes the product. If
Dive's output is used to guide care — a clinician acting on a predicted decline
risk, a recommendation influencing treatment — it is very likely a medical
device under UKCA / MHRA and EU MDR. If it is used strictly for research it
generally is not. The classification determines whether you need a quality
management system, clinical evaluation, post-market surveillance and a notified
body, and it is far cheaper to answer now than after a sale.

Dive's design helps here in one specific way: every prediction carries its
provenance, its validation status and its refusal reasons, and the models
abstain rather than guess. That is the evidence trail a regulator asks for.

## Reporting a vulnerability

Provide a private disclosure channel and a response commitment before first
customer deployment. A public issue tracker is not one.
