# DEPLOY.md — Pasang SUPERAGENT v4.0 di OpenClaw (dari nol)

Panduan deploy *agent*-nya ke runtime OpenClaw di VPS lo. (Beda dari `skills/hermes/references/deploy.md` yang itu soal deploy *smart contract*.)

> Catatan: ini **menggantikan** instruksi quick-start lama di README (`~/.openclaw/workspace/superagent-v3/` + `openclaw-agents.json`) yang udah gak sesuai konvensi OpenClaw sekarang. Pakai langkah di bawah.

---

## 0. Prasyarat

```bash
# OpenClaw udah keinstall & jalan (gateway)
openclaw --version

# Python 3.10+ dan pip
python3 --version
```

---

## 1. Taruh workspace

OpenClaw nge-inject bootstrap file (AGENTS.md, SOUL.md, IDENTITY.md, USER.md, TOOLS.md) dari **root workspace**. Jadi jangan nest di subfolder — arahin workspace langsung ke folder `openclaw/` ini.

```bash
# salin folder openclaw ke lokasi tetap
cp -r openclaw ~/superagent-v3

# init config OpenClaw kalau belum
openclaw setup           # bikin ~/.openclaw/openclaw.json
```

Edit `~/.openclaw/openclaw.json`, arahin workspace ke folder tadi:

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/superagent-v3"
    }
  }
}
```

Sekarang AGENTS.md/SOUL.md/IDENTITY.md/USER.md/TOOLS.md ada di root workspace → ke-inject otomatis tiap sesi.

---

## 2. Install dependency Python

```bash
cd ~/superagent-v3

# inti Hermes (wallet, swap, web3, dll)
pip install web3 eth-account mnemonic solders solana httpx \
            cryptography pysui aptos-sdk tonsdk base58 bip-utils \
            ens hyperliquid-python-sdk websockets

# governor/mev/integrity/memory/reflection/alerts/briefing/triage/vault/
# watchdog/contract_reader/writer/deploy → cukup web3+httpx+cryptography+stdlib (udah ke-cover)
```

Opsional (per fitur):

```bash
# browser automation (m15/browser.md)
pip install playwright && playwright install chromium

# voice input — STT lokal gratis (m15/multimodal)
pip install faster-whisper

# crypto dev: compile/test/deploy kontrak (v4.0)
curl -L https://foundry.paradigm.xyz | bash && foundryup
```

---

## 3. Isi config

```bash
cp .env.example .env
nano .env                 # isi HERMES_MASTER_PW [WAJIB] + RPC + key yang lo pakai
```

Edit `USER.md` (masih template) — nama, honorific, timezone, bahasa, domain fokus. Ini ke-inject tiap sesi, jadi isi dulu sebelum jalan.

Set spend cap governor di `.env` (sangat disarankan):

```bash
HERMES_MAX_TX_USD=500
HERMES_DAILY_CAP_USD=2000
HERMES_SESSION_CAP_USD=1000
```

---

## 4. Lock integritas skill (langkah operator)

Generate manifest di sumber tepercaya (mesin lo), supaya tampering kedeteksi:

```bash
python tools/skill_integrity.py generate            # tulis SKILLS.lock
# opsional, tandatangani:
# export HERMES_SIGNING_KEY=~/.hermes/sign.pem
# python tools/skill_integrity.py generate --sign
```

Verifikasi kapan pun:

```bash
python tools/skill_integrity.py verify              # exit 0 = bersih
```

---

## 5. Restart & cek

```bash
pm2 restart openclaw      # atau: systemctl restart openclaw / screen -r
```

Cek boot sequence jalan (lihat AGENTS.md): inject identity → TIME → m0 registry → USER → MEMORY → integrity verify → reflection cycle → briefing if due.

Tes cepat dari channel (Telegram/chat):

```
"siapa lo?"            → harusnya jawab sebagai SUPERAGENT
"cek gas ethereum"     → trigger web3 (m10/H-skill)
"baca kontrak 0x...."  → trigger H9 contract reader
```

---

## 6. Jadwalkan service (opsional)

```bash
# alert engine (poll terus)        → jalanin sebagai service/background task
# daily briefing (sekali/hari)     → cron, contoh jam 7 WIB:
0 7 * * *  cd ~/superagent-v3 && python -c "..."   # wire push_briefing ke notifier lo
# watchdog (mantau proses)         → service interval 30s
```

---

## Checklist ringkas

```
☐ openclaw setup → workspace = ~/superagent-v3
☐ pip install deps inti (+ playwright/whisper/foundry sesuai fitur)
☐ cp .env.example .env → isi HERMES_MASTER_PW + RPC + caps governor
☐ edit USER.md
☐ python tools/skill_integrity.py generate
☐ restart openclaw → tes "siapa lo?"
☐ (opsional) jadwalkan briefing/alert/watchdog
```

---

## Troubleshooting

- **Bootstrap file gak ke-inject** → workspace nunjuk ke subfolder, bukan root. AGENTS.md/SOUL.md harus di root workspace.
- **`integrity verify` exit 1 di VPS** → wajar kalau lo edit file pasca-generate; audit (m11) lalu `generate` ulang. Kalau lo gak ngedit apa-apa → investigasi, JANGAN jalanin operasi on-chain.
- **RPC publik timeout** → wajar (rate-limited). Isi `RPC_EVM_*` dengan endpoint lo / pakai RPCRouter failover.
- **`forge`/`playwright`/`whisper` not found** → fitur terkait kasih pesan install, sisanya tetap jalan.
- **Update config gak kebaca** → restart gateway OpenClaw.
