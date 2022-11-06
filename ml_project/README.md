# Magauiya Zhussip - Homework 1 on MLOps

## Checklist
- [x] Overall description of tech. decisions
- [x] Make checklist for PR
- [x] EDA and model prototype
- [x] Implement training script
- [x] Implement test script (inputs: model ckpt, testset, save path)
- [x] Module-based structure
- [x] Add logging
- [x] Test coverage: train, predict, etc.
- [x] Generate synthetic tests
- [x] Use config files (at least 2), additional points for using hydra
- [x] Use dataclasses, not dict
- [x] Freeze all dependencies
- [ ] Setup CI tests
- [ ] MlFlow: setup
- [ ] MlFlow: logging metrics     
- [ ] MlFlow: use Model Registry and add screenshot of run DVC usage
- [ ] MlFlow: choose entrypoints
- [ ] MlFlow: dataset versioning
- [ ] MlFlow: dvc pipeline   

## Proposed Solution 
As a solution for the heart disease classification, I chose LightGBM model. LightGBM is one of the SOTA in
terms of table data fitting. However, boosting models suffer from overfitting, therefore, I decreased depth, number of leafs,
and estimators. Also, training data is processed such that highly relied features are dropped. By this, we reach 80% accuracy 
compared to 71% accuracy of a default LightGBM.

## Installation
```
pip3 install -r requirements.txt
```

## Project Structure
![structure](./folder_structure.png)

## Train script
To train the classifier, run following command:
```
python3 main.py
```
To change the config file, please go to the ```main.py```.

## Test script
To inference the model, please you following command:
```
python3 main.py params.inference=True path.savefile='path to predictions' test.ckpt='path to checkpoint'
```

## Unittests
To run unittests:
```
python3 -m unittest discover ./tests
```

