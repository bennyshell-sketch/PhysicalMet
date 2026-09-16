# Hosting Dive

Running Dive on a server, so that people reach it in a browser and nothing has
to be installed on their machine. This is the route that sidesteps locked-down
Windows builds entirely: the laptop only needs a browser.

Everything below assumes a small Linux virtual machine — 2 vCPU and 4 GB of
memory is enough for a cohort of a few hundred — and a domain name you control.

---

## 1. What you are deploying

Two containers:

| Container | Job |
|---|---|
| `dive` | The application. Listens on port 8501 **on the internal network only**. |
| `proxy` | Caddy. Terminates TLS, obtains and renews the certificate, and is the only thing that can reach `dive`. |

The application container publishes no port to the host, so there is no way to
reach Dive except through the proxy over HTTPS. Dive additionally refuses any
browser connection whose `Host` header is not your domain, which is what stops
someone pointing their own hostname at your server's IP address.

All state lives on one Docker volume mounted at `/data`: accounts, the audit
log, uploaded workbooks, trained models, ingested batches. Nothing is stored
inside the container, so upgrading is a rebuild, not a migration.

---

## 2. First deployment

### Prerequisites

* A Linux host with Docker Engine and the Compose plugin.
* A domain name — say `dive.example.org` — with an **A record pointing at the
  host's public IP address**. Do this first: Caddy asks Let's Encrypt for a
  certificate on startup, and that only works once DNS resolves.
* Ports 80 and 443 open inbound. Port 80 is needed for the certificate
  challenge, and Caddy redirects it to HTTPS afterwards.

### Get the code onto the host

```bash
git clone <your repository url> dive
cd dive
```

### Generate the secrets

```bash
cp .env.example .env

python3 -c "import secrets; print('DIVE_SESSION_SECRET=' + secrets.token_urlsafe(48))"
python3 -c "from dive.security import DataVault; print('DIVE_ENCRYPTION_KEY=' + DataVault.generate_key())"
```

Put the output into `.env`, along with your domain and an email address for
certificate-expiry warnings. `.env` is in `.gitignore`; keep it that way.

**Write the encryption key down somewhere else as well.** Uploaded cohort files
are encrypted with it, and if you lose it they cannot be recovered — that is
what "encrypted" means.

### Owner console key (optional)

The owner console is the screen that shows what the model has learnt, what
changed and when. It is off unless you configure a key hash:

```bash
python3 -c "from dive.owner import OwnerGate; g=OwnerGate(); k=g.generate_key(); print('KEY :', k); print('HASH:', g.hash_key(k))"
```

Keep the `KEY` (that is what you type into the app), and put the `HASH` in
`.env` as `DIVE_OWNER_KEY_HASH`. The key itself is never stored on the server,
so nobody with access to the host can read it out.

### Start it

```bash
docker compose up -d --build
docker compose logs -f dive
```

Compose refuses to start if a required secret is missing, rather than starting
with a default — a deployment that silently comes up insecure is worse than one
that does not come up.

### First sign-in

Open `https://your-domain`. On first run Dive creates an `admin` account and
shows its password **once**, on the sign-in page. Copy it, sign in, and change
it immediately — Dive forces this before it lets you reach anything else.

If you would rather set the first password yourself, put it in `.env` as
`DIVE_ADMIN_PASSWORD` before starting. It still has to be changed at first
sign-in, because a password sitting in a file on the server is one leaked file
away from being an open door.

Then open **Users** and create an account per person. Three roles:

| Role | Screens |
|---|---|
| `viewer` | Data, Group comparison, Privacy and security. No individual-level results, and no audit entries. |
| `analyst` | The above, plus Individual prediction, Trajectory, Recommendations, and Training and data. |
| `admin` | Everything, plus Users, the audit log itself, and the Owner console door — which still needs the owner key. |

Give people the least role that lets them do their job. Individual-level
predictions are identifiable clinical inferences; they are not summary
statistics and should not be handed out by default.

---

## 3. Day-to-day operation

```bash
docker compose logs -f dive     # what the app is doing
docker compose restart dive     # restart without losing data
docker compose down             # stop (the volume survives)
docker compose up -d --build    # after pulling new code
```

### Backups

Everything worth keeping is in the `dive-data` volume.

```bash
docker run --rm \
  -v dive_dive-data:/data:ro \
  -v "$PWD":/backup \
  alpine tar czf /backup/dive-backup-$(date +%F).tar.gz -C /data .
```

Take the backup off the host — a backup on the same machine protects you
against exactly one failure mode, and not the likely ones. Test a restore
before you need it; an untested backup is a hope, not a plan.

Note that the backup contains participant data. Encrypt it at rest and treat it
with the same care as the live system.

### Upgrading

```bash
git pull
docker compose up -d --build
```

The volume is untouched, so accounts, the audit log and loaded data survive.

---

## 4. What is protected, and what is not

### Protected by this setup

* **In transit** — TLS 1.2+ with a real certificate, HSTS, and the usual
  hardening headers. No plaintext HTTP.
* **Reachability** — the application port is not published; the proxy is the
  only route in; a wrong `Host` header is refused.
* **Sign-in** — scrypt password hashes, a 12-character minimum enforced by the
  server, lockout after five failures that survives a restart, uniform failure
  messages so usernames cannot be probed, and signed sessions that expire.
  Changing a password ends every session opened with the old one.
* **Uploaded cohort files** — AES-GCM encrypted on disk using
  `DIVE_ENCRYPTION_KEY`, and authenticated, so a tampered file is refused rather
  than half-read.
* **What leaves the server** — every outbound model call passes a single
  fail-closed egress guard, and each one is recorded in the audit log.
* **What happened** — a hash-chained audit log, so an edited or deleted entry
  breaks the chain and is detected.
* **Blast radius** — the container runs as an unprivileged user with
  `no-new-privileges`, and holds no data of its own.

### Not protected, and your job

* **Everything Dive derives from an upload** — trained models, ingested
  batches, the audit log — is written in the clear. Run the volume on an
  encrypted disk (LUKS, or your cloud provider's encrypted block storage).
  That covers every file rather than the subset Dive happens to write through
  the vault.
* **Anyone with root on the host, or in the `docker` group,** can read the
  secrets out of the container environment and the data off the volume. Limit
  who has shell access. This is not a Dive-specific weakness; it is what
  administrator access means.
* **Operating system and dependency patching.** Rebuild periodically — the
  base image gets security updates, and an unpatched host undoes everything
  above.
* **No MFA and no SSO.** For anything clinical, authenticate against your
  institution's identity provider instead of Dive's own accounts. `dive/accounts.py`
  is written to be replaced by that rather than extended towards it.
* **No rate limiting at the proxy.** Account lockout limits password guessing,
  but a flood of requests will still consume resources. Put Dive behind your
  institution's gateway, or add Caddy's rate-limit module, if it faces the open
  internet.
* **An independent penetration test and a threat model.** Nothing in this file
  substitutes for either.

### Before real patient data

* A DPIA under UK GDPR / DPA 2018, covering the model providers if you enable
  the evidence advisor.
* Information-governance approval, and for NHS data, the relevant DSPT and
  data-sharing agreements.
* A decision on medical-device classification. If Dive's output guides care
  rather than research, UKCA/MHRA and EU MDR obligations very probably apply,
  and that decision shapes everything downstream. Settle it early.

---

## 5. Hardening further

Worth doing if the deployment faces the open internet rather than a university
or NHS network:

* **Restrict who can reach it at all** — an IP allow-list at your institution's
  firewall, or a VPN, in front of the proxy. The strongest control available is
  not being reachable by strangers.
* **A read-only container filesystem** — add `read_only: true` and a `tmpfs`
  for `/tmp` to the `dive` service. Test it in staging first: Streamlit writes
  to a few paths under the home directory.
* **Log shipping** — send `/data/audit.jsonl` to append-only storage on a
  different host. A tamper-evident log on the same machine as the attacker is
  evidence they can delete; the hash chain tells you it happened, not what was
  in it.
* **Turn off what you do not use** — leave every provider API key unset and
  Dive makes no external calls at all. The evidence advisor falls back to its
  curated library.

---

## 6. Troubleshooting

**The certificate does not issue.** Check DNS resolves to this host
(`dig +short your-domain`), and that port 80 is reachable from outside. Caddy
retries; `docker compose logs proxy` says what it is waiting for.

**The page loads but stays blank, or says "please wait".** That is the
websocket being refused. Check `DIVE_PUBLIC_HOSTNAME` in `.env` exactly matches
the domain in the address bar — no `https://`, no port, no trailing dot.

**"DIVE_SESSION_SECRET is not set" on the sign-in page.** The container started
without it. Check `.env` and `docker compose up -d` again. Until it is set,
everyone is signed out whenever the container restarts.

**Locked out of the admin account.** Lockout clears after 15 minutes. If the
password itself is lost, stop the stack, delete `accounts.json` from the volume,
and start again — Dive bootstraps a new admin. Everything else on the volume is
kept.

```bash
docker compose down
docker run --rm -v dive_dive-data:/data alpine rm /data/accounts.json
docker compose up -d
```

**Out of memory during training.** The `dive` service is capped at 4 GB in
`docker-compose.yml`. Raise it if the host has more.
