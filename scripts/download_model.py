"""Download author-published ModelScope files with a pinned per-file manifest.

No SDK/cache copy is needed. Files are resumed and SHA256-verified before rename.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from urllib.parse import quote


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mib-per-second", type=float, default=40,
                        help="Per-worker download rate limit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "relaymoe-download-manifest.json"
    if manifest.exists():
        locked = json.loads(manifest.read_text())
        if locked["model"] != args.model:
            raise ValueError("Destination belongs to a different model")
    else:
        url = f"https://modelscope.cn/api/v1/models/{args.model}/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        if data["Code"] != 200:
            raise RuntimeError(data)
        files = [f for f in data["Data"]["Files"] if f["Type"] == "blob"
                 and (f["Path"].endswith((".safetensors", ".json", ".txt", ".model"))
                      or f["Path"] in {"LICENSE", "README.md"})]
        locked = {"model": args.model, "files": files, "source": url}
        manifest.write_text(json.dumps(locked, indent=2))
    files = locked["files"]
    remaining = sum(max(0, f["Size"] - (args.output / f["Path"]).stat().st_size)
                    if (args.output / f["Path"]).exists() else f["Size"] for f in files)
    if shutil.disk_usage(args.output).free < remaining + (20 << 30):
        raise RuntimeError("Insufficient space including a 20 GiB reserve")
    print(json.dumps({"model": args.model, "files": len(files),
                      "bytes": sum(f["Size"] for f in files)}, ensure_ascii=False), flush=True)

    def fetch(f):
        rel = Path(f["Path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        dest = args.output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size == f["Size"] and sha256(dest) == f["Sha256"]:
                print("VERIFIED " + str(rel), flush=True)
                return
            raise RuntimeError(f"Existing file failed verification: {dest}")
        part = dest.with_suffix(dest.suffix + ".part")
        url = f"https://modelscope.cn/models/{args.model}/resolve/{f['Revision']}/{quote(f['Path'])}"
        for attempt in range(5):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset < f["Size"]:
                    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        append = offset > 0 and response.status == 206
                        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Invalid Content-Range")
                        start, downloaded, last_report = time.monotonic(), 0, time.monotonic()
                        with part.open("ab" if append else "wb") as stream:
                            while chunk := response.read(4 << 20):
                                stream.write(chunk)
                                downloaded += len(chunk)
                                target = downloaded / (args.mib_per_second * (1 << 20))
                                delay = target - (time.monotonic() - start)
                                if delay > 0:
                                    time.sleep(delay)
                                if time.monotonic() - last_report >= 30:
                                    print(f"PROGRESS {rel} {stream.tell()}/{f['Size']}", flush=True)
                                    last_report = time.monotonic()
                if part.stat().st_size != f["Size"] or sha256(part) != f["Sha256"]:
                    raise RuntimeError(f"Size or SHA256 mismatch: {rel}")
                part.replace(dest)
                print("DONE " + str(rel), flush=True)
                return
            except Exception as error:
                print(f"RETRY {rel} attempt={attempt+1} {error}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, files))
    (args.output / "relaymoe-download-complete.json").write_text(json.dumps({
        "model": args.model, "verified_files": len(files), "sha256_verified": True,
    }, indent=2))
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
