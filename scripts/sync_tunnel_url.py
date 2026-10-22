"""Keep the Hugging Face Space gateway pointed at the current Cloudflare tunnel URL.

Run periodically (systemd timer). Reads the latest *.trycloudflare.com URL from the tunnel service's journal and, if
it differs from the Space's TARGET_URL variable, updates it (the Space restarts with the new target).

Usage: python scripts/sync_tunnel_url.py <hf-user>/<space-name>
"""
import re
import subprocess
import sys

from huggingface_hub import HfApi


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
    api = HfApi()
    current = api.get_space_variables(space).get("TARGET_URL")
    if current is None or current.value != url:
        api.add_space_variable(space, "TARGET_URL", url)
        print(f"updated TARGET_URL -> {url}")
    else:
        print(f"unchanged: {url}")


if __name__ == "__main__":
    main(sys.argv[1])
