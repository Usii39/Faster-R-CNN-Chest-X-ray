# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 13:14:15 2025

@author: 0125i
"""

import pandas as pd
import matplotlib.pyplot as plt

# 讀取訓練資料
df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')

# from PIL import Image
# img_path = r"C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\normImage\aortic_atherosclerosis_calcification\10 (2)d.dcm.jpg"
# image = Image.open(img_path).convert("RGB")

#%% =========================
# 基本資料檢視
# =========================
print("資料筆數:", len(df))
print("\n欄位資訊:")
print(df.info())

#%%=========================
# 疾病標籤分佈（Label Distribution）
# =========================
label_counts = df['category'].value_counts()

print("\n疾病類別分佈:")
print(label_counts)

#%% =========================
# 視覺化：長條圖
# =========================
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title("Disease Category Distribution")
plt.xlabel("Disease Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% =========================
# 視覺化：圓餅圖
# =========================
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title("Proportion of Disease Categories")
plt.axis('equal')
plt.show()

#%% =========================
# 影像解析度分析
# =========================
plt.figure(figsize=(10, 5))
plt.scatter(df['Width'], df['Height'], alpha=0.6)
# plt.title("Image Resolution Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n影像寬度統計:")
print(df['Width'].describe())

print("\n影像高度統計:")
print(df['Height'].describe())
