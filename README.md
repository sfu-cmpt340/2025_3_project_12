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


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── data
    ├── image_data                      # Real images (zipped) and metadata
    ├── sketch_data                     # Sketches (zipped), before augmentation
├── src                                 
    ├── augment_benign_sketches.py      # Data augmention (benign sketches)
    ├── augment_malignant_sketches.py   # Data augmentation (malignant sketches)
    ├── combined_evaluation.py          # Evaluate model trained on real images and sketches
    ├── combined_training.py            # Train model on real images and sketches
    ├── file_paths.py                   # Util to handle file paths
    ├── generate_sketch_data_split.py   # Generate sketch train/validation/test split
    ├── model.py                        # Load base model (Inception V3)
    ├── real_image_dataset_loader.py    # Custom PyTorch DatasetLoader (real images)
    ├── real_image_dataset.py           # Custom PyTorch Dataset for real images
    ├── real_image_evaluation.py        # Evaluate model trained on real images
    ├── real_image_training.py          # Train model on real images
    ├── sketch_evaluation.py            # Evaluate model trained on sketches
    ├── sketch_training.py              # Train model on sketches
├── README.md
├── requirements.txt
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
