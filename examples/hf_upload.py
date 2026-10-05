"""hf_upload.py - upload a checkpoint to HF (stdlib only, v5.37g).

The durable fix for the 10-02 lesson: the upload tool lived ONLY in a
sandbox /tmp file, so the mid baseline never reached HF and was lost to
a local overwrite. This script ships IN the repo. Token from env:

    $env:HF_TOKEN = "hf_..."
    python examples/hf_upload.py checkpoints/mid_sft_v5.33.pt
    python examples/hf_upload.py checkpoints/mid_sft_v5.33.tok.json
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


INLINE_MAX = 1_000_000   # < 1MB: inline files[] (text/small artifacts);
                         # >= 1MB: LFS. Uploading a text README as an LFS
                         # pointer breaks the HF model card (found 10-03:
                         # README.md landed as an LFS pointer and the repo
                         # front page showed the pointer text)


def main(path, repo="chienhsinlin/helioslm", repo_path=None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN first ($env:HF_TOKEN='hf_...')")
    auth = {"Authorization": f"Bearer {token}"}
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    name = repo_path or ("checkpoints/" + os.path.basename(path))
    # inline ONLY for genuinely text-decodable files; binary files
    # (PDFs!) must use LFS regardless of size -- decoding a PDF with
    # errors="replace" corrupts it and the commit API 400s (found
    # 10-03 on the 313KB paper PDF)
    is_text = False
    try:
        data.decode("utf-8")
        is_text = True
    except UnicodeDecodeError:
        pass
    if is_text and len(data) < INLINE_MAX:
        payload = {"message": f"upload {name} (inline)",
                   "summary": f"upload {name}",
                   "description": f"inline upload of {name}",
                   "files": [{"path": name, "content": data.decode("utf-8")}],
                   "lfsFiles": []}
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{repo}/commit/main",
            method="POST", data=json.dumps(payload).encode(),
            headers={**auth, "Content-Type": "application/json",
                     "User-Agent": UA})
        commit = json.loads(urllib.request.urlopen(req, timeout=120).read())
        print("[inline] commit:", commit.get("commitUrl"), flush=True)
        print("DONE", flush=True)
        return
    print(f"[1/4] {name}: {len(data)} bytes sha={sha[:12]} [LFS]", flush=True)

    def api(url, payload=None, method="POST", headers=None, raw=None,
            timeout=600):
        body = raw if raw is not None else (
            json.dumps(payload).encode() if payload is not None else None)
        req = urllib.request.Request(url, method=method, data=body,
                                     headers=headers or {})
        return urllib.request.urlopen(req, timeout=timeout).read()

    api(f"https://huggingface.co/api/models/{repo}/preupload/main",
        {"files": [{"path": name, "size": len(data), "sha256": sha,
                    "sample": base64.b64encode(data[:8192]).decode()}]},
        headers={**auth, "Content-Type": "application/json", "User-Agent": UA})
    print("[2/4] preupload OK", flush=True)
    r = json.loads(api(
        f"https://huggingface.co/{repo}.git/info/lfs/objects/batch",
        {"operation": "upload", "transfers": ["basic"],
         "objects": [{"oid": sha, "size": len(data)}]},
        headers={**auth, "Content-Type": "application/vnd.git-lfs+json",
                 "Accept": "application/vnd.git-lfs+json", "User-Agent": UA}))
    obj = r["objects"][0]
    if "actions" in obj:
        up = obj["actions"]["upload"]
        req = urllib.request.Request(up["href"], method="PUT", data=data,
            headers={**up.get("header", {}), "User-Agent": UA,
                     "Content-Type": "application/octet-stream"})
        print("[3/4] PUT ...", flush=True)
        resp = urllib.request.urlopen(req, timeout=1800)
        print("    PUT status:", resp.status, flush=True)
    else:
        print("[3/4] object already on server", flush=True)
    commit = json.loads(api(
        f"https://huggingface.co/api/models/{repo}/commit/main",
        {"message": f"upload {name} (sha {sha[:12]})",
         "summary": f"upload {name}",
         "description": f"checkpoint artifact {name}",
         "files": [],
         "lfsFiles": [{"path": name, "oid": sha, "size": len(data)}]},
        headers={**auth, "Content-Type": "application/json", "User-Agent": UA}))
    print("[4/4] commit:", commit.get("commitUrl"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1],
         repo_path=(sys.argv[2] if len(sys.argv) > 2 else None))
