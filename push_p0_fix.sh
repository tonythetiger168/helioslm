#!/bin/bash
# HeliosLM P0 Phase — 一鍵推送腳本
# 用法: 將此檔案放在 helioslm 倉庫根目錄，執行: bash push_p0_fix.sh

set -e

echo "🚀 HeliosLM P0 Phase v1.0.2 推送腳本"
echo "=========================================="

# 檢查是否在 git 倉庫
if [ ! -d ".git" ]; then
    echo "❌ 錯誤: 當前目錄不是 git 倉庫"
    exit 1
fi

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo "⚠️  警告: Docker 未安裝，沙箱功能將無法使用"
fi

# 下載並解壓修復包
echo "📥 下載 P0 修復包..."
curl -L -o /tmp/helioslm_p0_fix_v1.0.2.zip \
    "https://kimi-files.oss-cn-beijing.aliyuncs.com/p0/helioslm_p0_fix_v1.0.2.zip" \
    2>/dev/null || echo "⚠️  請手動下載 zip 並放到 /tmp/"

if [ -f "/tmp/helioslm_p0_fix_v1.0.2.zip" ]; then
    unzip -q /tmp/helioslm_p0_fix_v1.0.2.zip -d /tmp/
    cp -r /tmp/helioslm_p0_fix/* .
    echo "✅ 檔案已覆蓋"
else
    echo "❌ 未找到修復包，請手動複製檔案"
    exit 1
fi

# 創建分支
echo "🌿 創建分支 p0-hardening-v1.0.2..."
git checkout -b p0-hardening-v1.0.2 2>/dev/null || git checkout p0-hardening-v1.0.2

# 提交
echo "📝 提交變更..."
git add -A
git commit -m "[P0] Security + Architecture Hardening v1.0.1 -> v1.0.2

- FIX: Docker sandbox replaces raw subprocess.run() (CVE-level RCE)
- FIX: Corrected Ultra config: 4.2T->2.8T total, 213B->35B active
- FIX: Real SentencePiece tokenizer replaces char-based placeholder
- FIX: KVCache for O(L) inference (was O(L^2))
- FIX: Dynamic sparsity now actually used in MoE routing
- FIX: Correct speculative decoding algorithm (Leviathan 2022)
- ADD: Hardened Docker sandbox image
- ADD: GitHub Actions CI (lint/test/security)
- ADD: P0_FIX_REPORT.md with migration guide"

# 推送
echo "⬆️  推送到 origin..."
git push origin p0-hardening-v1.0.2

echo ""
echo "=========================================="
echo "✅ 完成！請到 GitHub 創建 Pull Request:"
echo "   https://github.com/tonythetiger168/helioslm/pull/new/p0-hardening-v1.0.2"
echo ""
echo "📋 PR 標題建議:"
echo "   [P0] Security + Architecture Hardening v1.0.2"
echo ""
echo "🏷️  合併後創建 tag:"
echo "   git tag -a v1.0.2 -m 'P0 Hardening release'"
echo "   git push origin v1.0.2"
echo "=========================================="
