This repository provides code used for my bachelor thesis __"The best way to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods"__. 

[[_TOC_]]

## Directory structure

```
|--src
|  |--dl_models.py
|  |--prepare_data.py
|  |--test.py
|  |--train.py
|  |--R_scripts
|  |  |--feature_extraction.R
|  |  |--data_analysis.R
|--data
|  |--Stroop
|  |  |--DL
|  |  |  |--train
|  |  |  |  |--xpos.csv
|  |  |  |  |--ypos.csv
|  |  |  |  |--label.csv
|  |  |  |  |--undersampled_xpos.csv
|  |  |  |  |--undersampled_ypos.csv
|  |  |  |  |--undersampled_label.csv
|  |  |  |  |--oversampled_xpos.csv
|  |  |  |  |--oversampled_ypos.csv
|  |  |  |  |--oversampled_label.csv
|  |  |  |--test
|  |  |  |  |--xpos.csv
|  |  |  |  |--ypos.csv
|  |  |  |  |--label.csv
|  |  |--ML
|  |  |  |--train_data.csv
|  |  |  |--test_data.csv
|  |  |  |--undersampled_lrain_data.csv
|  |  |  |--undersampled_ltest_data.csv
|  |  |  |--oversampled_train_data.csv
|  |  |  |--oversampled_test_data.csv
|  |--Seminar
|  |--Pilot_Study
|  |--preprocessed
|  |--raw
|--trained_models
|  |-ML
|  |-DL
|--config.json
|--README.md
|--main.py
```

* `data/raw`: collected data through the pilot study and the main experiment.
* `data/preprocessed`: preprocessed data using the `feature_extraction.R` script.
* `trained_models`: trained ML and DL models are saved.

* In this thesis, following ML and DL algorithms were used:
    - ML candidates: Naive Bayes (NB), Linear Desciminant Analysis (LDA), K-Nearest Neighbors (KNN), Random Forest (RF) and Bagged Tree (TB).
    - DL candidates: CNN, GCN, DeepConvLSTM and Transformer-encoder-based-classifier 
* The implementation of the DL models can be found in `dl_models.py`.

## Usage

### Defining the configurations

* In the root directory, a configuration file is provided (`conf.json`). First, you need to define the necessary configurations for the `main/py`.
* `conf.json` consists of three main configurations:
    1. First one is `option`. If you will train ML and DL models on the data, define `option` as `train`. If you want to test trained models, define it as `test`.
    2. Second one is `balanced`. This parameter is used to define if balanced data should be used for training (You can choose between `undersampled` and `oversampled`). If you want to train models on imbalanced data, put `false` in this configuration.
    3. Third one is `model_type`. If you want to train the ML models, define it as `ML`. If you want to train the DL models, define it as `DL`. 
* Save the changes in `conf.json`. Now your configurations are ready for training and testing models.

### Running `main.py`

* After defining the configurations, run `python3 main.py`.
* You can follow the logs on the terminal.
* At the end, there will be a folder named as `results`. In this folder, the results (accuracy, balanced accuracy and F1 score for each model) will be saved. 
* If you trained ML or DL models, the trained models will be saved in the `trained_models` folder.

## Author

Tamaki Ogura (ogura@cl.uni-heidelberg.de)