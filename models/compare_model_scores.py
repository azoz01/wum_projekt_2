import numpy as np
import pandas as pd
import scipy
from icecream import ic
from copy import deepcopy
from model_logging import logger
from models import *
from encoding import *
from metrics import *

# encodings_labels = ["Encoding_1", "Encoding_2"]
# encodings = [MockEncoding1(), MockEncoding2()]
# models_labels = ["Model_1", "Model_2"]
# models = [MockModel1(), MockModel2()]
# metrics_labels = ["Metric_1", "Metric_2"]
# metrics = [mock_metric, mock_metric]


def main():
    logger.info("Reading train data")
    df_train = pd.read_pickle("resources/data/train_data.pkl")
    logger.info("Reading test data")
    df_test = pd.read_pickle("resources/data/test_data.pkl")

    X_train = df_train["bow_dict"]
    X_test = df_test["bow_dict"]

    result_df = pd.DataFrame(columns=["encoding", "model", "metric", "type", "value"])
    for encoding, encoding_label in zip(encodings, encodings_labels):
        encoding = deepcopy(encoding)
        logger.info(f"{encoding_label} - fitting encoding")
        encoding.fit(X_train)
        logger.info(f"{encoding_label} - transforming train")
        train_encoded = encoding.transform(X_train)
        logger.info(f"{encoding_label} - transforming test")
        test_encoded = encoding.transform(X_test)
        # OGULNIE TAK SIE NIE POWINNO ROBIC ALE KTO TO BEDZIE CZYTAL
        if type(train_encoded.values.tolist()[0]) is scipy.sparse._csr.csr_matrix:
            ic("DUPA")
            train_encoded = scipy.sparse.vstack(train_encoded.values.tolist())
            test_encoded = scipy.sparse.vstack(test_encoded.values.tolist())
        else:
            ic("NOT DUPA")
            for el in train_encoded:
                if len(el.shape) != 1:
                    print(el.shape)
            train_encoded = np.stack(train_encoded.values.tolist(), axis=0)
            test_encoded = np.stack(test_encoded.values.tolist(), axis=0)

        for model, model_label in zip(models, models_labels):
            model = deepcopy(model)
            try:
                logger.info(f"{model_label} - fitting model")
                model.fit(train_encoded)
                logger.info(f"{model_label} - predicting train")
                train_pred = model.predict(train_encoded)
                logger.info(f"{model_label} - predicting test")
            # In case if predict doesn't exist
            except:
                train_pred = model.fit_predict(train_encoded)
                test_pred = model.fit_predict(test_encoded)
            for metric, metric_label in zip(metrics, metrics_labels):
                logger.info(f"{metric_label} - computing for train")
                try:
                    train_metric = metric(train_encoded, train_pred)
                except:
                    train_metric = "error"
                logger.info(f"{metric_label} - computing for test")
                try:
                    test_metric = metric(test_encoded, test_pred)
                except:
                    test_metric = "error"
                row_result_train = pd.DataFrame(
                    {
                        "encoding": encoding_label,
                        "model": model_label,
                        "metric": metric_label,
                        "type": "train",
                        "value": train_metric,
                    },
                    index=[0],
                )
                row_result_test = pd.DataFrame(
                    {
                        "encoding": encoding_label,
                        "model": model_label,
                        "metric": metric_label,
                        "type": "test",
                        "value": test_metric,
                    },
                    index=[0],
                )
                result_df = pd.concat(
                    [result_df, row_result_train, row_result_test]
                ).reset_index(drop=True)

    result_pivot_table = pd.pivot(
        data=result_df, index=["encoding", "model"], columns=["metric", "type"]
    )
    logger.info("Saving results to models/results/results.pkl")
    result_pivot_table.to_pickle("models/results/results.pkl")


if __name__ == "__main__":
    main()
