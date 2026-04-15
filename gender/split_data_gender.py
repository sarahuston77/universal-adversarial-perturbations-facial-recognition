# Run after running setup_data.set
# Creates train_df, test_df, and copies files from data/crop_part1 into train_utk_dataset/ and test_utk_dataset
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = "data"
source_dir = os.path.join(data_dir, "crop_part1")
female_dir = "female_dataset"
male_dir = "male_dataset"

os.makedirs(female_dir, exist_ok=True)
os.makedirs(male_dir, exist_ok=True)

rows = []

# ---- Step 1: Parse filenames ----
for fname in os.listdir(source_dir):
  print(fname)
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

males = df[df["gender"] == 0]
females = df[df["gender"] == 1]

# ---- Step 4: Move files ----
for fname in males["filename"]:
  print(fname)
  shutil.move(os.path.join(source_dir, fname), os.path.join(male_dir, fname))

for fname in females["filename"]:
  shutil.move(os.path.join(source_dir, fname), os.path.join(female_dir, fname))

print()
print("Sanity checks, the gender proportions for both train and test should be similar ")
print("Males:")
print(males["gender"].value_counts(normalize=True))
print()
print("test:")
print(females["gender"].value_counts(normalize=True))
