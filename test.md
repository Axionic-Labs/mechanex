# Mechanex - Dev Setup Guide

## Prerequisites
- Backend must be running before login (local or staging)

## Install

```bash
cd "/Users/adityapatane/axionic labs/mechanex"
pip install -e .
```

## base URL Options

| Environment | URL |
|---|---|
| Local | `http://localhost:3000` |
| Staging | `https://axionic-mvp-backend-594546489999.us-east4.run.app` |

## Setup Steps

### New Users (`mechanex signup` auto-saves api_key to config)

```bash
# 1. Set base_url before signup (backend must be running)
mkdir -p ~/.mechanex && echo '{"base_url": "base_url"}' > ~/.mechanex/config.json

# 2. Signup — auto-creates account, logs in, and saves api_key to config
mechanex signup

# 3. Verify
mechanex whoami
```

### Existing Users

```bash
# 1. Set config BEFORE login (login merges JWT tokens into this file)
cat > ~/.mechanex/config.json << 'EOF'
{
  "api_key": "your_api_key_here",
  "base_url": "bsae_url"
}
EOF

# 2. Login — merges JWT tokens into existing config
mechanex login

# 3. Verify
mechanex whoami
```

## Notes
- Never use `cat > ~/.mechanex/config.json` after login — it overwrites JWT tokens
- `mechanex signup` auto-saves the api_key; `mechanex create-api-key` does not
- `base_url` must be set before login since login calls the backend
