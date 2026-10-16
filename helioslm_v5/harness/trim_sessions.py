#!/usr/bin/env python3
"""一次性瘦身 ~/.helios_ui/sessions/*.json：每個 session 只保留最近 100 則訊息。
超過 5MB 的檔案才處理；原檔自動備份到 sessions_backup_trim/。
用法: python trim_sessions.py
"""
import glob
import json
import os
import shutil
import time

KEEP = 100
MIN_BYTES = 5 * 1024 * 1024  # 5MB

sdir = os.path.expanduser(os.path.join("~", ".helios_ui", "sessions"))
mdir = os.path.expanduser(os.path.join("~", ".helios_ui", "media"))
bdir = sdir.rstrip(os.sep) + "_backup_trim"
os.makedirs(bdir, exist_ok=True)

total_before = total_after = 0
for p in sorted(glob.glob(os.path.join(sdir, "*.json"))):
    size = os.path.getsize(p)
    total_before += size
    if size < MIN_BYTES:
        continue
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    n = len(s.get("messages", []))
    if n <= KEEP:
        total_after += size
        continue
    shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
    s["messages"] = s["messages"][-KEEP:]
    s["updated"] = time.time()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    new_size = os.path.getsize(p)
    total_after += new_size
    print("trimmed %s: %d -> %d msgs, %.1fMB -> %.1fMB"
          % (os.path.basename(p), n, KEEP, size / 1e6, new_size / 1e6))

# media 目錄順便回報（TTS/影片會累積，不自動刪）
media_total = sum(os.path.getsize(os.path.join(mdir, f))
                  for f in os.listdir(mdir)) if os.path.isdir(mdir) else 0
print("sessions: %.1fMB -> %.1fMB | media: %.1fMB (備份在 %s)"
      % (total_before / 1e6, total_after / 1e6, media_total / 1e6, bdir))
