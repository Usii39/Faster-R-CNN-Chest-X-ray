# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 13:08:03 2025

@author: 0125i
增加bounding box的四個位置值
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops


def mask_to_bbox(mask_path, min_area=3000):
    """
    將二值化 mask 影像轉換為 bounding box 座標

    Parameters
    ----------
    mask_path : str
        Mask 影像檔案路徑（JPG / PNG）
    min_area : int
        最小保留的連通區域面積（預設 3000）

    Returns
    -------
    xmin, ymin, xmax, ymax : int
        Bounding box 座標

    Raises
    ------
    FileNotFoundError
        當 mask 檔案不存在
    ValueError
        當沒有符合條件的有效區域
    """

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    try:
        # 讀取影像並轉為 numpy array
        img = Image.open(mask_path)
        mask_np = np.array(img)

        # 建立 binary mask（非零為 True）
        binary_mask = mask_np > 0

        # 標記連通區域（8-connected）
        labeled_mask = label(binary_mask, connectivity=2)

        # 取得所有連通區域屬性
        regions = regionprops(labeled_mask)

        # 過濾面積過小的區域
        valid_regions = [r for r in regions if r.area >= min_area]

        if len(valid_regions) == 0:
            raise ValueError(f"No valid region found in mask: {mask_path}")

        # 取第一個符合條件的區域
        region = valid_regions[0]

        # region.bbox = (ymin, xmin, ymax, xmax)
        ymin, xmin, ymax, xmax = region.bbox

        return xmin, ymin, xmax, ymax

    except Exception as e:
        raise RuntimeError(f"Failed to process mask {mask_path}: {str(e)}")




def add_bbox_columns(df, base_path, category_col='category', mark_col='MarkPath'):
    """
    批次處理 mask 並將 bounding box 寫入 DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        原始資料表
    base_path : str
        mask 檔案的根目錄
    category_col : str
        類別欄位名稱（預設 'category'）
    mark_col : str
        mask 路徑欄位名稱（預設 'MarkPath'）

    Returns
    -------
    df : pandas.DataFrame
        新增 xmin, ymin, xmax, ymax 欄位後的 DataFrame
    """

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for idx, row in df.iterrows():
        try:
            category = row[category_col]

            if category == "normal":
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
            else:
                # mask_path = os.path.join(base_path, row[mark_col])
                mask_path = base_path+row[mark_col]
                xmin, ymin, xmax, ymax = mask_to_bbox(mask_path)

        except Exception as e:
            # 發生錯誤時，bbox 設為 0，並顯示警告
            print(f"[Warning] Index {idx}: {e}")
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)

        # 顯示進度
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} samples")

    # 新增欄位
    df = df.copy()
    df['xmin'] = xmin_list
    df['ymin'] = ymin_list
    df['xmax'] = xmax_list
    df['ymax'] = ymax_list

    return df


df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')
base_path = 'C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/train'

train2 = add_bbox_columns(df, base_path, category_col='category', mark_col='MarkPath')
train2.to_csv('C:/上課講義/碩二上/醫學影像DL/hw5/hwk05_data/train.csv',encoding='utf-8',index=False)


