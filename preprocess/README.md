# Preprocess for MASS-SS3


## Requirements

- Python 3.6
- numpy 1.15.4
- scipy 1.1.0

## Usage

- You can use the DE_PSD.py to compute DE and PSD features of a set of data.

- Data preparation for MASS-SS3

1. Get the MASS-SS3 dataset, and extract useful data to save as .mat file.

    - Default path is /data/SS3/
    - XX-XX-XXXX-Data.mat: include variable 'PSG' (n * channels * t, t is the number of sampling points in 30 seconds)
    - XX-XX-XXXX-Label.mat: include variable 'label' (n * 5, one-hot)

2. Generate 31 fold data set, using subject-independent scheme.

    - Default feature storage path is /data/DE_PSD/
    - 31 fold data set is default stored as /data/XXXX.npz

    ```shell
    python process_SS3.py
    ```


