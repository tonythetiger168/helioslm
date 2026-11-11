"""Create a family-level intake manifest from pinned portfolio snapshots.

Assign source candidates before tokenization. Unknown mappings and unpaired
RTL stay out. Per-project synthesis and benchmark overlap are later gates.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


RTL_REPO = "open-ip-portfolio"
VIP_REPO = "open-vip-portfolio"
HDL_EXTENSIONS = {".sv", ".svh", ".v", ".vh"}
MAX_FILE_BYTES = 1 << 20


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def payload(z, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe member path: {name}")
    if z.getinfo(name).file_size > MAX_FILE_BYTES:
        raise ValueError(f"Oversize source member: {name}")
    return z.read(name)


def source_file(z, name):
    raw = payload(z, name)
    return {"path": name, "bytes": len(raw), "sha256": digest(raw),
            "content_sha256": digest(raw.replace(b"\r\n", b"\n").strip())}


def members(z):
    names = z.namelist()
    if len(set(names)) != len(names):
        raise ValueError("Archive has duplicate member names")
    return names


def read_catalog(ip, vip):
    ip_names = members(ip)
    vip_names = members(vip)
    rtl = defaultdict(list)
    for name in ip_names:
        path = PurePosixPath(name)
        if len(path.parts) == 3 and path.parts[1] == "rtl" and path.name.endswith("_top.sv"):
            rtl[normalized_name(path.stem.removesuffix("_top"))].append(name)
    catalog = next(name for name in vip_names if name.endswith("/vip_individual_portfolio.csv"))
    csv_text = payload(vip, catalog).decode("utf-8-sig")
    archives = {PurePosixPath(name).name: name for name in vip_names
                if len(PurePosixPath(name).parts) == 3
                and PurePosixPath(name).parts[1] == "vips" and name.endswith(".zip")}
    rows, excluded = [], []
    for row in csv.DictReader(io.StringIO(csv_text)):
        name = row["Protocol"]
        archive = row["VIP檔案"]
        aliases = [normalized_name(alias) for alias in row["open-ip 對應"].split("/")]
        if (row["License"] != "Apache-2.0" or archive not in archives
                or any(len(rtl[alias]) != 1 for alias in aliases)):
            excluded.append({"protocol": name, "archive": archive,
                             "mapping": row["open-ip 對應"],
                             "reason": "missing or ambiguous RTL mapping, archive or license"})
            continue
        tops = [rtl[alias][0] for alias in aliases]
        ip_sources = []
        for top in tops:
            ip_sources.append(source_file(ip, top))
            wrapper = top.removesuffix("_top.sv") + "_apb.sv"
            if wrapper in ip_names:
                ip_sources.append(source_file(ip, wrapper))
        archive_path = archives[archive]
        archive_raw = payload(vip, archive_path)
        with zipfile.ZipFile(io.BytesIO(archive_raw)) as nested:
            vip_sources = [source_file(nested, path) for path in members(nested)
                           if PurePosixPath(path).suffix.lower() in HDL_EXTENSIONS]
        if not vip_sources:
            raise ValueError(f"No HDL source in {archive}")
        rows.append({"protocol": name, "category": row["類別"],
                     "origin_family": row["來源家族"], "rtl": ip_sources,
                     "vip_archive": archive_path, "vip_archive_sha256": digest(archive_raw),
                     "vip": vip_sources})
    return rows, excluded, {path for paths in rtl.values() for path in paths}


def assign(rows):
    """Keep protocol families and identical design cores in one partition."""
    parent = list(range(len(rows)))

    def find(node):
        while node != parent[node]:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    first = {}
    for index, row in enumerate(rows):
        keys = [("category", row["category"])]
        origin = row["origin_family"]
        # These labels identify catalog release waves, not protocol families.
        if origin and not re.match(r"(?i)(new-|wave\d)", origin):
            keys.append(("origin_family", origin))
        keys.extend(("content", file["content_sha256"])
                    for file in row["rtl"])
        for key in keys:
            first_index = first.setdefault(key, index)
            parent[find(index)] = find(first_index)

    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    holdouts = [group for group in components.values()
                if any(row["protocol"] == "APB" for row in group)]
    if len(holdouts) != 1 or len(components) < 2:
        raise ValueError("Cannot reserve APB and also produce independent training candidates")
    for group in components.values():
        group_id = digest("\n".join(sorted(row["protocol"] for row in group)).encode())[:16]
        for row in group:
            row["family_group"] = group_id
            row["split"] = "eval" if group is holdouts[0] else "train_candidate"
    # Common VIP infrastructure can be byte-identical across otherwise unrelated
    # families. Keep eval self-contained and exclude its copies from training.
    held_out_content = {file["content_sha256"] for row in holdouts[0]
                        for file in row["rtl"] + row["vip"]}
    for row in rows:
        if row["split"] != "train_candidate":
            continue
        copies = [file for file in row["vip"] if file["content_sha256"] in held_out_content]
        row["vip"] = [file for file in row["vip"] if file["content_sha256"] not in held_out_content]
        if copies:
            row["excluded_eval_copies"] = copies
        if not row["vip"] or any(file["content_sha256"] in held_out_content for file in row["rtl"]):
            row["split"] = "quarantined"
    return len(components)


def write_partition(path, rows, ip, vip):
    """Write one raw record per pair, preserving file boundaries and provenance."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            with zipfile.ZipFile(io.BytesIO(payload(vip, row["vip_archive"]))) as nested:
                record = {"protocol": row["protocol"], "family_group": row["family_group"],
                          "split": row["split"], "vip_archive": row["vip_archive"],
                          "files": []}
                for repo, source, files in ((RTL_REPO, ip, row["rtl"]),
                                            (VIP_REPO, nested, row["vip"])):
                    for file in files:
                        raw = payload(source, file["path"])
                        if digest(raw) != file["sha256"]:
                            raise ValueError(f"Source changed: {file['path']}")
                        record["files"].append({**file, "repo": repo,
                                                "text": raw.decode("utf-8", errors="strict")})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    raw = path.read_bytes()
    return {"records": len(rows), "bytes": len(raw), "sha256": digest(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--apb-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    receipt = json.loads(args.apb_receipt.read_text(encoding="utf-8"))
    if receipt.get("direct") != "passed" or receipt.get("uvm") != "passed":
        parser.error("the APB direct and UVM runs must pass before creating the split")
    manifest = json.loads((args.snapshots / "manifest.json").read_text(encoding="utf-8"))
    for repo in (RTL_REPO, VIP_REPO):
        raw = (args.snapshots / f"{repo}.zip").read_bytes()
        if len(raw) != manifest[repo]["bytes"] or digest(raw) != manifest[repo]["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {repo}")
    with zipfile.ZipFile(args.snapshots / f"{RTL_REPO}.zip") as ip, \
         zipfile.ZipFile(args.snapshots / f"{VIP_REPO}.zip") as vip:
        rows, excluded, tops = read_catalog(ip, vip)
        groups = assign(rows)
        args.out.mkdir(parents=True, exist_ok=False)
        shards = {}
        for name, label in (("train-candidates.jsonl", "train_candidate"), ("eval.jsonl", "eval")):
            shards[name] = write_partition(args.out / name,
                                           [row for row in rows if row["split"] == label], ip, vip)
        for repo, archive in ((RTL_REPO, ip), (VIP_REPO, vip)):
            for name in members(archive):
                path = PurePosixPath(name)
                if len(path.parts) == 2 and path.name in {"LICENSE", "NOTICE"}:
                    (args.out / f"{repo}-{path.name}").write_bytes(payload(archive, name))
    paired = {file["path"] for row in rows for file in row["rtl"]}
    result = {"schema": "hdl-portfolio-family-intake/1", "snapshots": manifest,
              "apb_receipt_sha256": digest(args.apb_receipt.read_bytes()),
              "family_groups": groups, "shards": shards, "records": rows, "excluded_vip": excluded,
              "unpaired_rtl": sorted(tops - paired), "training_ready": False,
              "pending": ["project synthesis", "benchmark overlap", "token shards"]}
    with (args.out / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"family_groups": groups, "eval": sum(row["split"] == "eval" for row in rows),
                      "train_candidates": sum(row["split"] == "train_candidate" for row in rows),
                      "quarantined_pairs": sum(row["split"] == "quarantined" for row in rows),
                      "excluded_vip": len(excluded), "unpaired_rtl": len(tops - paired)}))


if __name__ == "__main__":
    main()
