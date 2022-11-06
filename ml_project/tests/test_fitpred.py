import os
import sys
import unittest
import omegaconf
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.append('../')
from main import Classifier


class TestClassifier(unittest.TestCase):
    cfg = omegaconf.OmegaConf.load('./configs/lgbm_conf.yaml')
    
    def test_train_fit_simple(self):
        self.cfg.path.data = "./dataset/synthetic_test1.csv"
        self.cfg.params.exp_name = "test_1"
        app = Classifier(self.cfg)
        app.build()
        acc = app.train()
        self.assertGreater(acc, 0.95)

    def test_train_fit_random(self):
        self.cfg.path.data = "./dataset/synthetic_test2.csv"
        self.cfg.params.exp_name = "test_2"
        app = Classifier(self.cfg)
        app.build()
        acc = app.train()
        self.assertTrue(0.45 <= acc <= 0.55)

    @classmethod
    def tearDownClass(cls):
        os.system("rm -rf ./logs/test*")
        os.system("rm -rf ./models/test*")
    
if __name__ == '__main__':
    unittest.main()
    
    
    
