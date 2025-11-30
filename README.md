# DermoAI
### Effects of Shifting Dependence from Real-Image to Sketch-Based inputs for Dermatoscopic Machine Learning Models
--------

Exploring how sketches can supplement real image data in AI dermatoscopic diagnosis.

Group Members:
1. Charles Lee
2. Matthew Pham
3. Ana Premovic
4. Hieu Tran
5. Temirlan Utarbayev


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EZYBBlqVyEpJns_VDBKNJXkBpVjuvgHr5LT3pMin_eeGWg) | [Slack channel](https://cmpt340fall2025.slack.com/archives/C09EQ09FUAK) | [Project report](https://www.overleaf.com/8449614954rvhznbvyswzx#7ebf41) |
|-----------|---------------|-------------------------|


- Timesheet: Time and tasks completed/participated for this project, per student.
- Slack channel: Private Slack project channel.
- Project report: Overleaf project report document.


## Table of Contents
1. [Demo Video](#demo)

2. [Project File Description](#project-files)

3. [Installation](#installation)

4. [Reproducing This Project](#repro)


<a name="demo"></a>
## 1. Demo Video

Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

<a name="project-files"></a>
## 2. Project File Description

We train a model on real images only, sketches only, then a combination of both. Each training phase can be evaluated independently for comparison. The sketch data has been augmented, zipped, and uploaded already. The training/evaluation scripts use the zipped version, but if you want to see the sketch augmentation process you can run the augmentation scripts. Details on all files below.

```bash
repository
├── data
    ├── image_data                      # Real images (zipped) and metadata
    ├── sketch_data                     # Sketches (zipped), before and after augmentation
├── src                                 
    ├── augment_benign_sketches.py      # Data augmention (benign sketches)
    ├── augment_malignant_sketches.py   # Data augmentation (malignant sketches)
    ├── combined_evaluation.py          # Evaluate model trained on real images and sketches
    ├── combined_training.py            # Train model on real images and sketches
    ├── file_paths.py                   # Util to handle file paths
    ├── generate_sketch_data_split.py   # Generate sketch train/validation/test split
    ├── model.py                        # Load base model (Inception V3)
    ├── real_image_dataset_loader.py    # Custom PyTorch DatasetLoader for real images
    ├── real_image_dataset.py           # Custom PyTorch Dataset for real images
    ├── real_image_evaluation.py        # Evaluate model trained on real images
    ├── real_image_training.py          # Train model on real images
    ├── sketch_evaluation.py            # Evaluate model trained on sketches
    ├── sketch_training.py              # Train model on sketches
├── README.md                           # You are here          
├── requirements.txt                    # Pip dependencies
```

<a name="installation"></a>

## 3. Installation

```bash
git clone $THISREPO
cd $THISREPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<a name="repro"></a>
## 4. Reproduction
```bash
cd $THISREPO/src
python3 real_image_training.py
python3 sketch_training.py
python3 combined_training.py
python3 real_image_evaluation.py
python3 sketch_evaluation.py
python3 combined_evaluation.py
```
View evaluation results in terminal. Read our report for detailed conclusions.
