#!/bin/zsh
# Save your Odds API key to .env (git-ignored) without showing it on screen.
cd "$(dirname "$0")/.."
read -s "k?Paste your key, then press Enter: "
echo
if [[ -z "$k" ]]; then
  echo "Nothing was pasted, so nothing was saved."
  exit 1
fi
echo "ODDS_API_KEY=$k" > .env
chmod 600 .env
echo "Saved."
