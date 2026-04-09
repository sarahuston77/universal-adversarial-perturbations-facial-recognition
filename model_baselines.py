import os
from tqdm import tqdm
from deepface import DeepFace

def parse_label(filename):
    # [age] is an integer from 0 to 116, indicating the age
    # [gender] is either 0 (male) or 1 (female)
    # [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)
    split = filename.split('_', 3)
    print(split)
    if len(split) != 4:
        age, gender, race = split[0], split[1], -1 # missing race, probably should delete these
    else:
        age, gender, race, _ = filename.split('_', 3)
        
    return int(age), int(gender), int(race)

def evaluate_gender(folder):
    total = 0
    correct = 0
    
    male_total = 0
    male_correct = 0
    
    female_total = 0
    female_correct = 0

    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]

    for fname in tqdm(files, desc="Processing images", unit="img"):
        if not fname.endswith(".jpg"):
            continue
        
        parsed = parse_label(fname)
        if parsed is None:
            continue
        
        _, true_gender, _ = parsed
        path = os.path.join(folder, fname)

        try:
            result = DeepFace.analyze(
                img_path=path,
                actions=["gender"],
                enforce_detection=False,
                detector_backend="skip"
            )
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        # DeepFace returns dict or list depending on version
        if isinstance(result, list):
            result = result[0]

        gender_dict = result["gender"]

        pred_gender_str = max(gender_dict, key=gender_dict.get)
        pred_gender = 0 if pred_gender_str.lower() == "man" else 1

        # Update stats
        total += 1
        
        if true_gender == 0:
            male_total += 1
            if pred_gender == true_gender:
                male_correct += 1
        else:
            female_total += 1
            if pred_gender == true_gender:
                female_correct += 1
        
        if pred_gender == true_gender:
            correct += 1

    return {
        "overall_accuracy": correct / total if total else 0,
        "male_accuracy": male_correct / male_total if male_total else 0,
        "female_accuracy": female_correct / female_total if female_total else 0,
        "counts": {
            "total": total,
            "male_total": male_total,
            "female_total": female_total
        }
    }

stats = evaluate_gender("test_utk_dataset/")

print("Overall accuracy:", stats["overall_accuracy"])
print("Male accuracy:", stats["male_accuracy"])
print("Female accuracy:", stats["female_accuracy"])
print("Counts:", stats["counts"])