"""publish_cloudflare.py — per-VRAM download bundles and page (v5.186).

  python examples/package_tiers.py --run runs/dense --out dist/
  python examples/publish_cloudflare.py --packages dist/ --out cf/ \\
      --base-url https://dl.example.com --version v5.186-pilot

Reads the tier packages in --packages (one directory per tier, as
package_tiers.py writes them) and writes:

  cf/r2/<version>/*.zip   one zip per tier and quantization, for R2
  cf/site/                index.html + packages.json, for Cloudflare Pages

Links point at --base-url (the R2 bucket's download domain), never at
GitHub. Upload with scripts/cloudflare_publish.sh cf/.
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.cloudflare import (  # noqa: E402
    build_site, bundle_package, check_base_url)
from helioslm_v5.src.export.package import TIERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", required=True, help="package_tiers.py --out directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True,
                    help="R2 download domain, e.g. https://dl.example.com")
    ap.add_argument("--version", required=True, help="e.g. v5.186-pilot")
    ap.add_argument("--status", choices=["preview", "release"], default="preview",
                    help="preview = pilot weights (tagged on the page)")
    ap.add_argument("--name", default="HeliosLM")
    args = ap.parse_args()
    check_base_url(args.base_url)
    out = Path(args.out)
    entries = []
    for tier in TIERS:
        pkg = Path(args.packages) / tier
        if not (pkg / "manifest.json").exists():
            print(tier, "no package, listed as in development (開發中)")
            continue
        e = bundle_package(pkg, out / "r2" / args.version, version=args.version,
                           status=args.status)
        entries.append(e)
        print(tier, [(b["zip"], b["bytes"]) for b in e["bundles"]])
    build_site(entries, out / "site", base_url=args.base_url, version=args.version,
               name=args.name)
    print("site:", out / "site")


if __name__ == "__main__":
    main()
