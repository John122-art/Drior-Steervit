#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || -z "$1" ]]; then
  echo "Usage: bash scripts/push_to_remote.sh <empty-remote-repository-url>" >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
remote_url="$1"
if git remote get-url origin >/dev/null 2>&1; then
  if [[ "$(git remote get-url origin)" != "$remote_url" ]]; then
    echo "Existing origin differs; refusing to replace it." >&2
    exit 1
  fi
else
  git remote add origin "$remote_url"
fi
git push --set-upstream origin main
