# universal-adversarial-perturbations-facial-recognition

## Dataset

We use a subset of the UTKFace dataset (~8000 images).

To set up:

```bash
bash setup_data.sh


---

### Optional Step: Make sure dataset is ignored in github

```bash
echo "data/" >> .gitignore
```

## Set-Up
```
pip install deepface
```

---
## Splitting Dataset into train and test
After running setup_data.sh, you'll have a folder called data/crop_part1. Now we want to split it into test and train datasets, but with equal proportions of gender and age.

To do this, run 
```
python split_data_test_train.py
```

If you run into an error about not having scikit installed run ```pip install -r requirements.txt```.

This should only take a couple minutes max, maybe even less. You'll now have two folders test_utk_dataset and train_utk_dataset. In the terminal output, you should see something like this:
```
Train: 6844, Test: 2934
Sanity checks, these next two values should be very similar:
gender
1    0.553039
0    0.446961
Name: proportion, dtype: float64
gender
1    0.552488
0    0.447512
Name: proportion, dtype: float64
```

1. To run your model on the entire dataset regardless of gender, age, etc., use ```files = os.listdir("train_utk_dataset")``` or ```files = os.listdir("test_utk_dataset")```, depending on if you're running train or test, in your code. Then run your model on ```files```.

2. To run your model on just the male faces, add ```files = [f for f in files if get_gender(f) == 0]``` after the above line. To run your model on just the female faces, add ```files = [f for f in files if get_gender(f) == 1]``` instead. 

In other words, to run on the entire training dataset:
```
files = os.listdir("train_utk_dataset")
```

To run on all female faces in the test dataset:
```
files = os.listdir("test_utk_dataset")
files = [f for f in files if get_gender(f) == 1]
```
