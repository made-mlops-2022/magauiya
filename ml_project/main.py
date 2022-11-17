"""
Author: Magauiya Zhussip
Course: MLOps
Homework: 1
"""
import os
import logging
import pickle
import hydra
import pandas as pd
from sklearn.metrics import classification_report as clf_report
from sklearn.model_selection import train_test_split


class Classifier:
    """
    Binary classifier for heart disease detection
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        """
        Initialization & create necessary dirs
        """
        # Make dirs
        self._make_dir()
        self._make_logger()
        # Init model
        self.logger.info(f"Model: {self.cfg.model._target_}")
        self.model = hydra.utils.instantiate(self.cfg.model)

    def _make_logger(self):
        self.logger = logging.getLogger("train")
        self.logger.setLevel(level=logging.DEBUG)

        logFileFormatter = logging.Formatter(
            fmt="%(levelname)s %(asctime)s Func: %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        filename = os.path.join(self.cfg.path.logs, f"train_{self.cfg.params.exp_name}.log")
        fileHandler = logging.FileHandler(filename=filename)
        fileHandler.setFormatter(logFileFormatter)
        self.logger.addHandler(fileHandler)

    def train(self):
        """
        Train script
        """
        self.logger.debug("Loading dataset ...")
        data = self._load_data(path=self.cfg.path.data)
        x_train, x_test, y_train, y_test = train_test_split(
            data.drop(["condition"], axis=1),
            data["condition"],
            test_size=self.cfg.data.split,
            random_state=self.cfg.data.seed,
        )

        self.logger.debug("Start training ...")
        self.model.fit(x_train, y_train)
        preds = self.model.predict(x_test)
        self.logger.info("Train finished!")
        self.logger.debug(clf_report(y_test, preds))

        acc = clf_report(y_test, preds, output_dict=True)["accuracy"]
        if acc < 0.50:
            self.logger.warning(
                f"Low accuracy: {acc} of {type(self.cfg.model).__name__} model"
            )
        self._save_ckpt()
        self.logger.info("Model saved!")
        return acc

    def _save_ckpt(self):
        save_path = os.path.join(
            self.cfg.path.models, self.cfg.params.exp_name + ".pkl"
        )
        with open(save_path, "wb") as file:
            pickle.dump(self.model, file)

    def _preprocess(self, data):
        drops = ["thalach", "chol", "age", "oldpeak", "trestbps", "thal"]
        columns = data.columns
        for col in drops:
            if col in columns:
                data = data.drop(columns=[col])
        return data

    def _load_data(self, path):
        data = pd.read_csv(path)
        if self.cfg.data.preprocess:
            data = self._preprocess(data)
        return data

    def _make_dir(self):
        os.makedirs(self.cfg.path.logs, exist_ok=True)
        os.makedirs(self.cfg.path.models, exist_ok=True)

    def test(self):
        """
        Inference script
        """
        with open(self.cfg.test.ckpt, "rb") as file:
            model = pickle.load(file)
        data = self._load_data(path=self.cfg.test.data)
        data = self._preprocess(data)
        preds = model.predict(data)
        pd.DataFrame(preds).to_csv(self.cfg.path.savefile)


@hydra.main(version_base=None, config_path="configs", config_name="lgbm_conf.yaml")
def main(cfg):
    """
    Run binary classifier with configs
    """
    app = Classifier(cfg)
    if not cfg.params.inference:
        app.build()
        _ = app.train()
    else:
        app.test()


if __name__ == "__main__":
    main()
