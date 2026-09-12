# HeliosLM GitHub Push Guide

## Option 1: SSH (Recommended - Already Set Up)

Your SSH key has been generated. Add this public key to GitHub:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGjncf/LdFSpqpqeg3AQkU0OStvJ9CJxEjO0lbGFJz7N helioslm@ai.local
```

### Steps:
1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Paste the key above
4. Click "Add SSH key"
5. Then run:

```bash
cd /mnt/agents/output
git remote add origin git@github.com:YOUR_USERNAME/helioslm.git
git branch -M main
git push -u origin main --force
```

## Option 2: HTTPS + Personal Access Token

### Steps:
1. Generate token: https://github.com/settings/tokens/new
   - Select scopes: `repo` (full control)
   - Expiration: 30 days (or no expiration)
2. Create empty repo: https://github.com/new
   - Name: `helioslm`
   - Do NOT initialize with README
3. Run the interactive script:

```bash
cd /mnt/agents/output
./push.sh
```

Then enter:
- Your GitHub username
- Repository name (default: helioslm)
- Your Personal Access Token

## Option 3: One-Shot Command (if you have token ready)

```bash
cd /mnt/agents/output
export GH_USER="your_username"
export GH_TOKEN="ghp_xxxxxxxxxxxx"
export GH_REPO="helioslm"

# Create repo
curl -H "Authorization: token ${GH_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"${GH_REPO}\",\"private\":false}"

# Push
git remote add origin https://${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git
git branch -M main
git push -u origin main --force
```

## Verification

After push, visit: `https://github.com/YOUR_USERNAME/helioslm`

You should see:
- 295+ files
- 32,862 lines of code
- README.md rendered
- All v4.1 + v5.0 modules

## Repository Stats

| Metric | Value |
|--------|-------|
| Files | 295 |
| Code lines | 32,862 |
| Commits | 1 |
| Phases | v4.1 + v5.0 + Phase3/4/5 |
| Modules | P0-P8 + Deep-Dive |
