# 🩻 Chest X-Ray Multi-Lesion Detection System
> 基於改進型 Faster R-CNN 之胸腔 X 光 多重病灶自動偵測系統

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-green.svg)

## 📖 專案簡介 (Introduction)
本專案旨在開發一套自動化的胸腔 X 光病灶偵測系統，能夠同時識別並定位包含心臟肥大、肺浸潤、脊椎側彎等 7 種常見胸腔疾病。針對醫學影像常見的「資料稀缺」、「類別極度不平衡」與「病灶尺度變異大」等挑戰，本專案在傳統 Faster R-CNN 架構上進行了深度優化，並結合 EigenCAM 進行了模型決策的視覺化與錯誤分析。

## ✨ 核心技術亮點 (Technical Highlights)
* **模型升級**：採用 `ResNet-50 FPN V2` 作為 Backbone，並載入最新 COCO 預訓練權重，提升基礎特徵提取能力。
* **自適應錨框 (Custom Anchors)**：捨棄通用設定，針對胸腔病灶幾何特徵重新設計 Anchor。
  * **Sizes**: `(16, 32, 96, 224, 584)`，完整覆蓋從微小結節到大範圍心臟肥大的尺度。
  * **Aspect Ratios**: 擴充至 `(0.25, 0.5, 1.0, 2.0, 3.0)`，精準捕捉脊椎側彎 (0.25) 與肋膜增厚等極端細長/寬扁病灶。
* **類別平衡策略 (Class Balancing)**：實作 `WeightedRandomSampler` 搭配替換採樣機制，強制提升稀有病灶 (如肋膜增厚) 在訓練批次中的曝光率。
* **線上光度增強 (Online Photometric Augmentation)**：引入 Color Jitter ($\pm 50\%$ 亮度與對比)，防止模型對過採樣之稀有樣本產生過度擬合。
* **可解釋性分析 (XAI)**：整合 `EigenCAM` 進行熱力圖視覺化，深入探討模型的捷徑學習 (Shortcut Learning) 與紋理識別限制。

## 📊 實驗結果 (Results)
本模型使用 COCO 評估標準進行驗證，在 IoU=0.5 的標準下取得穩健的偵測表現：

| Metric | Value | Description |
| :--- | :--- | :--- |
| **mAP @ IoU=0.50** | `0.365` | 整體平均精準度 |
| **AR @ MaxDets=100** | `0.487` | 平均召回率 (低漏診率) |
| **AP (Scoliosis)** | `0.612` | 表現最佳類別 (具備明顯幾何結構) |

> 📌 **深度洞察 (Insights):** > 透過 EigenCAM 分析發現，模型在結構明確的病灶 (如脊椎側彎) 表現優異，但面對紋理模糊的病灶 (如肺野浸潤) 容易產生邊界不確定性。此外，在極端稀有類別中觀察到模型依賴次要特徵 (如脊椎) 進行推斷的現象，為未來的改進提供了明確方向。

