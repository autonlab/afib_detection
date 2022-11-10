# afib_detection
## Prediction
see `./notebooks` directory

## Detection
## Data store
Once logged into lab server navigate to `/zfsmladi/originals`, and within you'll find a large store of h5 files.
Find the functions you can use to pull, read, and manipulate these files within the `/data` directory as seen below.

**Note** If you'd like to read from remote data on your local machine check out our [faq repo](https://github.com/autonlab/auton-faqs/blob/main/howTos/how-to-ssh.md) in the **Mounting remote disks** section

## Dev setup
```
python -m venv {your virtual environment name}
source ./{your virtual environment name}/bin/activate
pip install -r requirements.txt
```

## Repository contents
Below you'll see a directory overview with potential files of interest explained
```
.
├── featurize.py
├── train.py
├── test.py
├── analysis.py
├── requirements-dev.txt
├── requirements.txt
├── data
│   ├── assets                  # store for data worth remembering
│   ├── config.yml
│   ├── computers.py            # data computation
│   ├── manipulators.py         # data manipulation
│   ├── stratified_sample.py    # functions for collecting segments from mladi store
│   └── utilities.py            # utilities for finding and manipulating h5 source files
├── model
│   ├── config.yml
│   ├── assets
│   ├── labelmodel.py
│   ├── mitbih.py
│   └── utilities.py
└── results
    └── assets                  # storage for output plots/artifacts
```