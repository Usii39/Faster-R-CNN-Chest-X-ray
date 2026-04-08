# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:00:27 2025

@author: 0125i
"""

#%%
import pandas as pd
df = pd.read_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\train.csv')
print(len(df['ID'].unique()))
print(len(df['Filename'].unique()))


# %%
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

#步驟 5.1：以「病人 ID」為單位的多標籤分層分割
def split_by_patient_id(df, test_size=0.2, random_state=42):
    """
    以病人 ID 為單位進行多標籤分層分割，避免 data leakage
    """
    np.random.seed(random_state)
    
    # 1. 取得所有唯一病人 ID
    patient_ids = df['ID'].unique()
    
    # 2. 收集每位病人的所有 class_id（多標籤）
    patient_labels = []
    for pid in patient_ids:
        labels = df.loc[df['ID'] == pid, 'class_id'].unique().tolist()
        patient_labels.append(labels)
    
    # 3. Multi-label one-hot encoding
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(patient_labels)
    
    # X 必須是 2D array
    X = np.array(patient_ids).reshape(-1, 1)
    
    # 4. Iterative stratified split
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X, y, test_size=test_size
    )
    
    train_patient_ids = X_train.flatten()
    val_patient_ids = X_val.flatten()
    
    # 5. 根據病人 ID 篩選 DataFrame
    train_df = df[df['ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_df = df[df['ID'].isin(val_patient_ids)].reset_index(drop=True)
    
    # 6. 印出統計資訊
    print("=== Split Statistics ===")
    print(f"Total patients      : {len(patient_ids)}")
    print(f"Train patients      : {len(train_patient_ids)}")
    print(f"Validation patients : {len(val_patient_ids)}")
    print(f"Train samples       : {len(train_df)}")
    print(f"Validation samples  : {len(val_df)}\n")
    
    print("Class distribution (Train):")
    print(train_df['category'].value_counts(), "\n")
    
    print("Class distribution (Validation):")
    print(val_df['category'].value_counts())
    
    return train_df, val_df


train_df, val_df = split_by_patient_id(df)
train_df.to_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\trainDf.csv',index=False)
val_df.to_csv(r'C:\上課講義\碩二上\醫學影像DL\hw5\hwk05_data\valDf.csv',index=False)



#%%
# 步驟 5.2：視覺化與驗證分割結果
import matplotlib.pyplot as plt

def visualize_and_validate_split(train_df, val_df, label_column='category'):
    """
    驗證病人 ID 分割正確性，並視覺化 train/val 類別分布
    """
    # ---------- 功能 1：驗證病人 ID 不重疊 ----------
    train_ids = set(train_df['ID'].unique())
    val_ids = set(val_df['ID'].unique())
    intersection = train_ids.intersection(val_ids)
    
    print("=== Patient ID Validation ===")
    print(f"Train patients : {len(train_ids)}")
    print(f"Val patients   : {len(val_ids)}")
    print(f"Total samples : {len(train_df) + len(val_df)}")
    
    if len(intersection) == 0:
        print("✔ 病人 ID 無重疊，未發生 data leakage\n")
    else:
        raise ValueError("✘ 發現病人 ID 重疊，請檢查分割流程")
    
    # ---------- 功能 2：類別分布視覺化 ----------
    train_counts = train_df[label_column].value_counts().sort_index()
    val_counts = val_df[label_column].value_counts().sort_index()
    
    categories = sorted(set(train_counts.index).union(val_counts.index))
    train_vals = [train_counts.get(cat, 0) for cat in categories]
    val_vals = [val_counts.get(cat, 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_vals, width, label='Train')
    plt.bar(x + width/2, val_vals, width, label='Validation')
    
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel("Number of Samples")
    plt.xlabel("Category")
    plt.title("Class Distribution: Train vs Validation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ---------- 功能 3：比例統計表 ----------
    print("=== Class Distribution Table ===")
    total_train = len(train_df)
    total_val = len(val_df)
    
    stats = []
    for cat in categories:
        t_cnt = train_counts.get(cat, 0)
        v_cnt = val_counts.get(cat, 0)
        stats.append([
            cat,
            t_cnt, f"{t_cnt / total_train:.2%}",
            v_cnt, f"{v_cnt / total_val:.2%}"
        ])
    
    stats_df = pd.DataFrame(
        stats,
        columns=["Category", "Train Count", "Train %", "Val Count", "Val %"]
    )
    
    print(stats_df)
    
    return stats_df

stats_df = visualize_and_validate_split(train_df, val_df)
