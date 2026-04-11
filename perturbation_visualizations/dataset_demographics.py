"""
Analyzes and visualizes the demographic breakdown (gender, race, age)
of the train and test UTK datasets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

TRAIN_FOLDER = "../train_utk_dataset"
TEST_FOLDER  = "../test_utk_dataset"

GENDER_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS   = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
AGE_BINS      = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
AGE_BIN_LABELS = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]

def parse_filename(fname):
    """Returns (age, gender, race) or None if unparseable."""
    parts = fname.split("_", 3)
    if len(parts) < 3:
        return None
    try:
        age    = int(parts[0])
        gender = int(parts[1])
        race   = int(parts[2])
        return age, gender, race
    except ValueError:
        return None

def load_dataset(folder):
    ages, genders, races = [], [], []
    skipped = 0
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            skipped += 1
            continue
        age, gender, race = parsed
        ages.append(age)
        genders.append(gender)
        races.append(race)
    return np.array(ages), np.array(genders), np.array(races), skipped

def print_stats(name, ages, genders, races, skipped):
    n = len(ages)
    print(f"\n{'='*50}")
    print(f"  {name}  (n={n}, skipped={skipped})")
    print(f"{'='*50}")

    print("\nGender:")
    for k, label in GENDER_LABELS.items():
        count = np.sum(genders == k)
        print(f"  {label:10s}: {count:5d}  ({count/n:.1%})")

    print("\nRace:")
    for k, label in RACE_LABELS.items():
        count = np.sum(races == k)
        print(f"  {label:10s}: {count:5d}  ({count/n:.1%})")

    print("\nAge bins:")
    bin_indices = np.digitize(ages, AGE_BINS) - 1
    for i, label in enumerate(AGE_BIN_LABELS):
        count = np.sum(bin_indices == i)
        print(f"  {label:6s}: {count:5d}  ({count/n:.1%})")
    print(f"\n  Mean age: {ages.mean():.1f}, Median: {np.median(ages):.0f}, Std: {ages.std():.1f}")

print("Loading datasets...")
train_ages, train_genders, train_races, train_skip = load_dataset(TRAIN_FOLDER)
test_ages,  test_genders,  test_races,  test_skip  = load_dataset(TEST_FOLDER)

print_stats("TRAIN", train_ages, train_genders, train_races, train_skip)
print_stats("TEST",  test_ages,  test_genders,  test_races,  test_skip)

# ── Plotting ──────────────────────────────────────────────────────────────────
datasets = [
    ("Train", train_genders, train_races, train_ages),
    ("Test",  test_genders,  test_races,  test_ages),
]

# --- Gender ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("UTK Dataset — Gender: Train vs. Test", fontsize=13, fontweight="bold")
for col, (name, genders, races, ages) in enumerate(datasets):
    n = len(ages)
    ax = axes[col]
    labels = [GENDER_LABELS[k] for k in sorted(GENDER_LABELS)]
    counts = [np.sum(genders == k) for k in sorted(GENDER_LABELS)]
    bars = ax.bar(labels, [c/n for c in counts], color=["#4C72B0","#DD8452"])
    ax.set_title(f"{name} (n={n})")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{count}\n({count/n:.1%})", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("demographics_gender.png", dpi=150, bbox_inches="tight")
print("Saved to demographics_gender.png")
plt.close()

# --- Race ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("UTK Dataset — Race: Train vs. Test", fontsize=13, fontweight="bold")
for col, (name, genders, races, ages) in enumerate(datasets):
    n = len(ages)
    ax = axes[col]
    labels = [RACE_LABELS[k] for k in sorted(RACE_LABELS)]
    counts = [np.sum(races == k) for k in sorted(RACE_LABELS)]
    bars = ax.bar(labels, [c/n for c in counts],
                  color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"])
    ax.set_title(f"{name} (n={n})")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{count}\n({count/n:.1%})", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("demographics_race.png", dpi=150, bbox_inches="tight")
print("Saved to demographics_race.png")
plt.close()

# --- Age ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("UTK Dataset — Age: Train vs. Test", fontsize=13, fontweight="bold")
for col, (name, genders, races, ages) in enumerate(datasets):
    n = len(ages)
    ax = axes[col]
    bin_indices = np.digitize(ages, AGE_BINS) - 1
    counts = [np.sum(bin_indices == i) for i in range(len(AGE_BIN_LABELS))]
    bars = ax.bar(AGE_BIN_LABELS, [c/n for c in counts], color="#4C72B0")
    ax.set_title(f"{name} (n={n}, mean age={ages.mean():.1f})")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(c/n for c in counts) * 1.25)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{count}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("demographics_age.png", dpi=150, bbox_inches="tight")
print("Saved to demographics_age.png")
plt.close()
