# Run after running setup_data.set
# Creates train_df, test_df, and copies files from data/crop_part1 into train_utk_dataset/ and test_utk_dataset
import os
import sys
import sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

source_dir = sys.argv[1]
white_dir = sys.argv[2]
asian_dir = sys.argv[3]
other_dir = sys.argv[4]
indian_dir = sys.argv[5]
black_dir = sys.argv[6]

os.makedirs(white_dir, exist_ok=True)
os.makedirs(asian_dir, exist_ok=True)
os.makedirs(other_dir, exist_ok=True)
os.makedirs(indian_dir, exist_ok=True)
os.makedirs(black_dir, exist_ok=True)

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

white = df[df["race"] == 0]
black = df[df["race"] == 1]
asian = df[df["race"] == 2]
indian = df[df["race"] == 3]
other = df[df["race"] == 4]
print(white)

# ---- Step 4: Move files ----
def moveFiles(df, dirName):
  for fname in df["filename"]:
    shutil.move(os.path.join(source_dir, fname), os.path.join(dirName, fname))

moveFiles(white, white_dir)
moveFiles(black, black_dir)
moveFiles(asian, asian_dir)
moveFiles(indian, indian_dir)
moveFiles(other, other_dir)
