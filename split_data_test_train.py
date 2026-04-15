# Run after running setup_data.set, takes in data_dir, source_dir, train_di
# Creates train_df, test_df, and copies files from data/crop_part1 into train_utk_dataset/ and test_utk_dataset
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

source_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]

if not os.listdir(source_dir):
    raise ValueError("Source directory is empty. Did you already run this script?")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

rows = []

# ---- Step 1: Parse filenames ----
for fname in os.listdir(source_dir):
    if not fname.endswith(".jpg"):
        continue

    parts = fname.split("_")
    if len(parts) < 4:
        continue

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except:
        continue

    rows.append({
        "filename": fname,
        "age": age,
        "gender": gender,
        "race": race
    })

df = pd.DataFrame(rows)

# ---- Step 2: Create age buckets ----
def age_bucket(age):
    if age < 10: return "0-9"
    elif age < 20: return "10-19"
    elif age < 30: return "20-29"
    elif age < 40: return "30-39"
    elif age < 50: return "40-49"
    elif age < 60: return "50-59"
    else: return "60+"

df["age_bucket"] = df["age"].apply(age_bucket)

# Combine gender + age bucket
df["stratify"] = df["age_bucket"] + "_" + df["gender"].astype(str)

# ---- Step 3: Stratified split ----
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["stratify"]
)

# ---- Step 4: Move files ----
for fname in train_df["filename"]:
    shutil.move(os.path.join(source_dir, fname), os.path.join(train_dir, fname))

for fname in test_df["filename"]:
    shutil.move(os.path.join(source_dir, fname), os.path.join(test_dir, fname))

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

print()
print("Sanity checks, the gender proportions for both train and test should be similar ")
print("train:")
print(train_df["gender"].value_counts(normalize=True))
print()
print("test:")
print(test_df["gender"].value_counts(normalize=True))

# for doing just male/female dataset, you can filter in code
# entire dataset: files = os.listdir("train_utk_dataset")
# male: files = [f for f in files if get_gender(f) == 0]
# female: files = [f for f in files if get_gender(f) == 1]
