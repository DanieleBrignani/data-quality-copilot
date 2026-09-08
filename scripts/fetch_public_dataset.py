"""Download the real public dataset used as the project's second demo case.

Usage::

    python scripts/fetch_public_dataset.py [--output data/public]

The file is committed to the repository, so this script exists to refresh it rather
than to make the demo depend on network access at a conference stand. It writes the
bytes exactly as served: the encoding defects in the source file are the point, and
"helpfully" re-encoding them would destroy the case the dataset is here to make.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "public"

#: CKAN package for "Elenco attività storiche e di tradizione nel Comune di Milano".
PACKAGE_ID = "624e5fff-9e40-4adc-9b82-a5d06fa95a26"
PACKAGE_URL = f"https://dati.comune.milano.it/api/3/action/package_show?id={PACKAGE_ID}"
FILENAME = "milano_attivita_storiche.csv"
USER_AGENT = "dqcopilot-demo/0.1 (+https://github.com/)"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return bytes(response.read())


def _csv_resource(package: dict) -> dict:
    """Return the package's CSV resource.

    Raises:
        RuntimeError: If the package no longer publishes a plain CSV.
    """
    for resource in package.get("resources", []):
        url = str(resource.get("url", ""))
        if str(resource.get("format", "")).upper() == "CSV" and url.endswith(".csv"):
            return dict(resource)
    raise RuntimeError("The package no longer offers a plain CSV resource.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    package = json.loads(_get(PACKAGE_URL))["result"]
    resource = _csv_resource(package)
    raw = _get(str(resource["url"]))

    destination = args.output / FILENAME
    destination.write_bytes(raw)

    metadata = {
        "title": package.get("title"),
        "publisher": package.get("maintainer") or package.get("organization", {}).get("title"),
        "licence": package.get("license_title"),
        "licence_url": package.get("license_url"),
        "package_url": f"https://dati.comune.milano.it/dataset/{PACKAGE_ID}",
        "resource_url": resource["url"],
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (args.output / "source.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {destination} ({len(raw):,} bytes)")
    print(f"  licence : {metadata['licence']}")
    print(f"  sha256  : {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
