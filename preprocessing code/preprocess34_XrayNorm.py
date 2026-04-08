# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:58:41 2025

@author: 0125i
"""

import os
import numpy as np
import pydicom
import pandas as pd

#%% preprocess3



def X_ray_normalization(dcm_path, vmin=0, vmax=2.5):
    """
    對單張 X-ray DICOM 影像進行正規化處理

    Parameters
    ----------
    dcm_path : str
        DICOM 檔案路徑
    vmin : float
        Linear stretching 下界（預設 0）
    vmax : float
        Linear stretching 上界（預設 2.5）

    Returns
    -------
    origin : np.ndarray
        原始影像像素陣列
    log_img : np.ndarray
        經 Intensity Log-Transformation 後的影像
    normalize_img : np.ndarray
        最終正規化後、範圍在 [0, 1] 的影像
    """

    if not os.path.exists(dcm_path):
        raise FileNotFoundError(f"DICOM file not found: {dcm_path}")

    try:
        # 讀取 DICOM
        dcm = pydicom.dcmread(dcm_path)

        # 取得原始像素值
        origin = dcm.pixel_array.astype(np.float32)

        # 讀取必要 DICOM metadata
        WW = float(dcm.WindowWidth)
        WC = float(dcm.WindowCenter)
        bits_stored = int(dcm.BitsStored)

        # 計算 window 上下界
        imax = (WW + 2 * WC) / 2
        imin = (2 * WC - WW) / 2

        # 將像素值限制在 [imin, imax]
        x_clamped = np.clip(origin, imin, imax)

        # Intensity log-transformation
        log_img = -np.log((1.0 + x_clamped) / (2 ** bits_stored))

        # Saturation-based linear stretching
        normalize_img = (log_img - vmin) / (vmax - vmin)
        normalize_img = np.clip(normalize_img, 0.0, 1.0)

        return origin, log_img, normalize_img

    except Exception as e:
        raise RuntimeError(f"Failed to normalize DICOM {dcm_path}: {str(e)}")


import matplotlib.pyplot as plt
    
def visualize_normalization(dcm_path, vmin=0, vmax=2.5):
    """
    視覺化 X-ray 正規化前後的影像與像素分布

    Parameters
    ----------
    dcm_path : str
        DICOM 檔案路徑
    vmin : float
        Linear stretching 下界
    vmax : float
        Linear stretching 上界
    """

    # 執行正規化
    origin, log_img, normalize_img = X_ray_normalization(dcm_path, vmin, vmax)

    # 建立 1x3 子圖
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子圖 1：原始影像
    im0 = axes[0].imshow(origin, cmap='gray')
    axes[0].set_title('Original DICOM Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # 子圖 2：正規化後影像
    im1 = axes[1].imshow(normalize_img, cmap='gray')
    axes[1].set_title('Normalized Image')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # 子圖 3：log後影像
    # im2 = axes[2].imshow(log_img, cmap='gray')
    # axes[2].set_title('Log Image')
    # axes[2].axis('off')
    # plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    

df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')
base_path = 'C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/train'

dcm_path = base_path + df.loc[151,'ImagePath']
origin, log_img, normalize_img = X_ray_normalization(dcm_path, vmin=0, vmax=2.5)

visualize_normalization(dcm_path, vmin=0, vmax=2.5)


#%% preprocess4 儲存Normalized Image

def save_train_images(df, base_path, output_dir, vmin=0, vmax=2.5):
    """
    將訓練集 DICOM 影像正規化後儲存為 JPG，並依 category 建立子資料夾

    Parameters
    ----------
    df : pandas.DataFrame
        包含影像資訊的 DataFrame
    base_path : str
        DICOM 影像的根目錄
    output_dir : str
        輸出 JPG 影像的資料夾
    vmin, vmax : float
        X-ray 正規化使用的最小與最大值
    """

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():

        # image_id = row['ID']
        image_name = row['Filename']
        category = row['category']
        # 方法：使用 lstrip('/') 把左邊的斜線去掉，再進行 join
        dicom_path = os.path.join(base_path, row['ImagePath'].lstrip('/\\'))

        # 建立 category 子資料夾
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # X-ray 正規化
        _, _, image_norm = X_ray_normalization(dicom_path,vmin=vmin, vmax=vmax)

        # 輸出檔名
        # save_path = os.path.join(category_dir, f"{image_id}.jpg")
        save_path = os.path.join(category_dir, f"{image_name}.jpg")

        # 儲存為灰階 JPG
        plt.imsave(save_path, image_norm, cmap='gray')

        # 顯示進度
        if (idx + 1) % 50 == 0:
            print(f"已處理 {idx + 1} 張訓練影像")

    print("訓練集影像儲存完成")


def save_test_images(df, base_path, output_dir, vmin=0, vmax=2.5):
    """
    將測試集 DICOM 影像正規化後儲存為 JPG（不依 category 分類）

    Parameters
    ----------
    df : pandas.DataFrame
        包含影像資訊的 DataFrame
    base_path : str
        DICOM 影像的根目錄
    output_dir : str
        輸出 JPG 影像的資料夾
    vmin, vmax : float
        X-ray 正規化使用的最小與最大值
    """

    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    for idx, row in df.iterrows():

        # image_id = row['ID']
        image_name = row['Filename']
        dicom_path = os.path.join(base_path, row['ImagePath'].lstrip('/\\'))

        # X-ray 正規化
        _, _, image_norm = X_ray_normalization(dicom_path, vmin=vmin, vmax=vmax)

        # 輸出檔名
        # save_path = os.path.join(test_dir, f"{image_id}.jpg")
        save_path = os.path.join(test_dir, f"{image_name}.jpg")

        # 儲存為灰階 JPG
        plt.imsave(save_path, image_norm, cmap='gray')

        # 顯示進度
        if (idx + 1) % 50 == 0:
            print(f"已處理 {idx + 1} 張測試影像")

    print("測試集影像儲存完成")


df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')
base_path = 'C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/train'
output_dir = 'C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/normImage'
save_train_images(df, base_path, output_dir, vmin=0, vmax=2.5)

df2 = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\test.csv')
base_path = 'C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/test'
save_test_images(df2, base_path, output_dir, vmin=0, vmax=2.5)

#%% 4-2 train和test新增Normalized Image Path 以及 轉換df['category']為 label Encoding

df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')
output_dir = '/normImage'
# new_image_paths = [output_dir + '/' + df.loc[i,'category'] + '/' + df.loc[i,'ID'] + '.jpg' for i in df.index]
new_image_paths = [output_dir + '/' + df.loc[i,'category'] + '/' + df.loc[i,'Filename'] + '.jpg' for i in df.index]
df['NormImagePath'] = new_image_paths

df2 = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\test.csv')
output_dir = '/normImage/test/'
# new_image_paths = [output_dir + df2.loc[i,'ID'] + '.jpg' for i in df2.index]
new_image_paths = [output_dir  + '/' + df2.loc[i,'Filename'] + '.jpg' for i in df2.index]
df2['NormImagePath'] = new_image_paths


# 1. 依照「出現順序」抓出所有不重複的類別
# Pandas 的 unique() 會從上到下掃描，先看到的排前面
ordered_labels = df['category'].unique()

# 2. 建立對照表 (Dictionary)
# enumerate 會幫這些標籤依序配上 0, 1, 2...
label_map = {label: idx for idx, label in enumerate(ordered_labels)}

# (選用) 印出來確認一下，確保 'normal' 真的是對應到 0
print("生成的類別對照表：")
for label, idx in label_map.items():
    print(f"ID {idx}: {label}")
# 3. 將對照表套用到 DataFrame
df['class_id'] = df['category'].map(label_map)

df.to_csv('C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/train.csv',encoding='utf-8',index=False)
df2.to_csv('C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/test.csv',encoding='utf-8',index=False)

# img = Image.open(r"C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\normImage\test\002.dcm.jpg").convert("RGB")
# img.shape

