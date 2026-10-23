"""Keep the static Hugging Face Space pointed at the current Cloudflare tunnel URL.

Run periodically (systemd timer). Reads the latest *.trycloudflare.com URL from the tunnel service's journal and,
when it changes, uploads it as backend.json to the Space, whose page reads that file to find the API.

Usage: python scripts/sync_tunnel_url.py <hf-user>/<space-name>
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi

STATE = "cache/backend_url.txt"


def current_tunnel_url() -> str | None:
    log = subprocess.run(["journalctl", "--user", "-u", "geoai-tunnel", "--no-pager", "-n", "400"],
                         capture_output=True, text=True).stdout
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    return urls[-1] if urls else None


def main(space: str) -> None:
    url = current_tunnel_url()
    if not url:
        print("no tunnel URL found")
        return
    last = open(STATE).read().strip() if os.path.exists(STATE) else None
    if url == last:
        print(f"unchanged: {url}")
        return
    body = json.dumps({"url": url, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1)
    HfApi().upload_file(path_or_fileobj=body.encode(), path_in_repo="backend.json", repo_id=space,
                        repo_type="space", commit_message="Update backend URL")
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        f.write(url)
    print(f"updated backend.json -> {url}")


if __name__ == "__main__":
    main(sys.argv[1])
