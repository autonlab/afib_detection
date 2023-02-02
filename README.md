# afib_detection
## Data sets
- [Long Term Atrial Fibrillation DB (ltafdb)](https://physionet.org/content/ltafdb/1.0.0/): 84 patients who have 'paroxysmal or sustained' atrial fibrillation, with lengthy recordings often lasting a day continuously. Annotated beats as well as rhythms.
- [MIT-BIH Arrhythmia DB](https://physionet.org/content/mitdb/1.0.0/): 48 half-hour excertps from 47 patients, also annotated by beats as well as rhythms
- MLADI Subset: Stratified sample (to maintain demographic distribution) from mladi store (sample of which can be found on lab servers under `/zfsmladi/originals`). No annotations save the ones we've collected ourselves.

**MLADI Note**
Once logged onto lab server navigate to `/zfsmladi/originals`, and within you'll find a large store of h5 files.
Find the functions you can use to pull, read, and manipulate these files within the `/data` directory as seen below.

**General Note** If you'd like to read from remote data on your local machine check out our [faq repo](https://github.com/autonlab/auton-faqs/blob/main/howTos/how-to-ssh.md) in the **Mounting remote disks** section

## Dev setup
```
python -m venv {your virtual environment name}
source ./{your virtual environment name}/bin/activate
pip install -r requirements.txt
```

## Repository contents
A directory overview with files of interest explained:
```
.
├── analysis.py
├── featurize.py
├── train.py#code to train suite of models for detection and prediction tasks, should be split up into separate files
├── test.py
├── data
│   ├── assets                  # store for data worth caching
│   ├── config.yml
│   ├── computers.py            # data computation
│   ├── nk_computers.py
│   ├── manipulators.py         # data manipulation
│   ├── stratified_sample.py    # functions for collecting segments from mladi store
│   └── utilities.py            # utilities for finding and manipulating h5 source files
├── detection
│   └── application.py
├── kyle
│   ├── train_cnn_dg.py #trains convnet with [MMD regularization)(https://machine-learning-note.readthedocs.io/en/latest/math/MMD.html)
│   └── cv_nntransformer.py #trains transformer net, expects a given test fold index as runtime parameter
├── prediction
│   └── loadData.py
├── model
│   ├── config.yml
│   ├── assets
│   ├── auton_survival #git submodule
│   ├── df4tsc #git submodule
│   ├── mitbih.py
│   └── utilities.py
├── notebooks
│   ├── pairwiseLFs.ipynb
│   ├── restitching.ipynb #restitching ltafdb episodes at specified knit lengths
│   └── rollingAverageFeaturization.ipynb #normalization by patient baselines
├── requirements.txt
└── README.md
```