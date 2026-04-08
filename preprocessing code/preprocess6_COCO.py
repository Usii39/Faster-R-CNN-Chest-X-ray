# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 20:12:38 2025

@author: 0125i
"""

# 步驟 6.1：COCO 格式轉換函數
import json
import os
from datetime import datetime
import pandas as pd


def create_coco_json(df, categories_list, output_path):
    """
    將 DataFrame 轉換為 COCO detection 格式並儲存為 JSON
    """
    coco = {
        "info": {
            "description": "Chest X-ray Disease Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # ---------- categories（排除 normal） ----------
    category2id = {}
    cat_id = 1  # COCO category_id 通常從 1 開始
    for cat in categories_list:
        if cat == "normal":
            continue
        coco["categories"].append({
            "id": cat_id,
            "name": cat,
            "supercategory": "chest_disease"
        })
        category2id[cat] = cat_id
        cat_id += 1

    # ---------- images ----------
    image_id_map = {}
    for idx, row in enumerate(df.itertuples()):
        file_name = row.NormImagePath.lstrip('/')
        file_name_idx = file_name.find('/')
        file_name = file_name[file_name_idx+1:]
        image_id_map[file_name] = idx

        coco["images"].append({
            "id": idx,
            "file_name": file_name,
            "width": int(row.Width),
            "height": int(row.Height)
        })

    # ---------- annotations ----------
    ann_id = 0
    for row in df.itertuples():
        
        if row.category == "normal":
            continue

        file_name = row.NormImagePath.lstrip('/')
        file_name_idx = file_name.find('/')
        file_name = file_name[file_name_idx+1:]

        xmin, ymin, xmax, ymax = row.xmin, row.ymin, row.xmax, row.ymax
        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            continue

        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id_map[file_name],
            "category_id": category2id[row.category],
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0,
            "segmentation": []
        })
        ann_id += 1

    # ---------- save json ----------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)

    print(f"COCO JSON saved to: {output_path}")



# 步驟 6.2：批次產生訓練與驗證 COCO JSON
from collections import Counter

def generate_train_val_coco(train_df, val_df, categories_list, output_dir):
    """
    產生 train / val 的 COCO annotation JSON，並印出統計資訊
    """
    os.makedirs(output_dir, exist_ok=True)

    train_json_path = os.path.join(output_dir, "annotations_train.json")
    val_json_path = os.path.join(output_dir, "annotations_val.json")

    create_coco_json(train_df, categories_list, train_json_path)
    create_coco_json(val_df, categories_list, val_json_path)

    # ---------- statistics ----------
    def print_stats(df, name):
        non_normal_df = df[df['category'] != 'normal']
        print(f"\n=== {name} Statistics ===")
        print(f"Images        : {df['NormImagePath'].nunique()}")
        print(f"Annotations   : {len(non_normal_df)}")
        print("Category count:")
        for cat, cnt in Counter(non_normal_df['category']).items():
            print(f"  {cat:50s}: {cnt}")

    print_stats(train_df, "Training Set")
    print_stats(val_df, "Validation Set")

    return train_json_path, val_json_path


# 步驟 6.3：驗證 COCO JSON 正確性
def validate_coco_json(json_path):
    """
    驗證 COCO JSON 格式與內容正確性
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    required_keys = {"info", "images", "annotations", "categories"}
    if not required_keys.issubset(coco.keys()):
        print("❌ 缺少必要的 COCO keys")
        return False

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_ids = [img["id"] for img in images]
    ann_ids = [ann["id"] for ann in annotations]
    category_ids = {cat["id"] for cat in categories}

    # image_id 唯一且連續
    if sorted(image_ids) != list(range(len(image_ids))):
        print("❌ image_id 不連續或不唯一")
        return False

    # annotation_id 唯一
    if len(ann_ids) != len(set(ann_ids)):
        print("❌ annotation_id 不唯一")
        return False

    image_id_set = set(image_ids)

    for ann in annotations:
        # image_id 存在
        if ann["image_id"] not in image_id_set:
            print("❌ annotation image_id 不存在")
            return False

        # category_id 存在
        if ann["category_id"] not in category_ids:
            print("❌ annotation category_id 不存在")
            return False

        # bbox 格式
        bbox = ann["bbox"]
        if len(bbox) != 4 or any(v < 0 for v in bbox):
            print("❌ bbox 格式錯誤")
            return False

        # bbox 不超出影像範圍
        img = images[ann["image_id"]]
        x, y, w, h = bbox
        if x + w > img["width"] or y + h > img["height"]:
            print("❌ bbox 超出影像範圍")
            return False

    print(f"✔ COCO JSON 驗證通過: {json_path}")
    return coco


#%%

train_df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\trainDf.csv')
val_df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\valDf.csv')
categories_list = ['aortic_curvature',
                 'aortic_atherosclerosis_calcification',
                 'cardiac_hypertrophy',
                 'intercostal_pleural_thickening',
                 'lung_field_infiltration',
                 'degenerative_joint_disease_of_the_thoracic_spine',
                 'scoliosis']


train_json, val_json = generate_train_val_coco(
    train_df, val_df, categories_list, output_dir=r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\annotations'
)

train_coco = validate_coco_json(train_json)
val_coco = validate_coco_json(val_json)


#%%
print(train_coco['categories'])
print()
print(train_coco['images'][:5])
print()
print(train_coco['annotations'][:5])





