# Plan: v5.3 → v5.4 修復全部剩餘問題 + 進版

## Stage 1 — 全量掃尾審查(2 verifiers 並行)
- V1:代碼剩餘問題掃描——v5.2 審查 MINOR 清單中未修項、v5.3 修復引入的新問題(回歸)、v5.3 報告「剩餘限制」中可修項(如 decode 非單調 position_ids 支援、引擎異常安全、m9 generate eval 恢復等)
- V2:全量測試壓力驗證——兩套件、邊界輸入、文檔與代碼一致性最終核對

## Stage 2 — 修復(coders 按文件所有權分派,數量視 findings)
## Stage 3 — 最終驗證(兩套件全過)
## Stage 4 — 進版 v5.4:版本字符串、CHANGELOG、README,打包 helioslm_v5.4_full.zip + backups/,V5.4 報告
