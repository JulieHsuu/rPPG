# 地端 rPPG 血壓研究管線

這個目錄只用於研究與驗證，不會在模型通過驗證前把血壓放回產品介面。

## 已取得的官方材料

- `data/CLBP300_Dataset_Specifications.xlsx`：CLBP-300 官方欄位資料。
- `data/clbp300_signal_extraction.rar`：官方訊號擷取範例與權重。
- `data/clbp300_samples.rar`：官方公開的 5 位受試者影片，只能驗證流程，不能建立可用模型。

完整 CLBP-300 約 60 GB，需要向資料集作者申請並簽署 DUA；不得重新散布影片或影格。

## 執行

在 `backend` 目錄執行：

```powershell
.\.venv\Scripts\python.exe training\bp\extract_features.py training\bp\data\clbp300_sample\ClBP-300_samples --output training\bp\features.csv
.\.venv\Scripts\python.exe training\bp\train_model.py training\bp\features.csv
```

訓練程式會按受試者 ID 固定切分資料；少於 60 位受試者會只產生 `validation_report.json` 並拒絕啟用模型。正式驗證至少需要 15 位完全未出現在訓練集的測試受試者，且收縮壓、舒張壓的平均偏差與誤差標準差都必須通過門檻。

即使公開資料驗證通過，仍應以實際目標 Webcam 搭配合格袖帶式血壓計，收集同步資料進行裝置與個人校正。
