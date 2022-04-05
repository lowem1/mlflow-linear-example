import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV
import mlflow


dataset_file_loc: str = "./Credit_N400_p9.csv"

full_dataset: pd.DataFrame = pd.read_csv(filepath_or_buffer=dataset_file_loc)

full_dataset["Student"].replace({"Yes": 1, "No": 0}, inplace=True)

full_dataset["Married"].replace({"Yes": 1, "No": 0}, inplace=True)


full_dataset["Gender"].replace({"Female": 1, "Male": 0}, inplace=True)


feature_cols: list = [
    "Income",
    "Limit",
    "Rating",
    "Cards",
    "Age",
    "Education",
    "Gender",
    "Student",
    "Married",
]

response_col: str = "Balance"

standardized_features: pd.DataFrame = (
    full_dataset[feature_cols] - full_dataset[feature_cols].mean()
) / full_dataset[feature_cols].std()
standardized_response: pd.DataFrame = (
    full_dataset[response_col] - full_dataset[response_col].mean()
)  # centered
standardized_response_std: pd.DataFrame = (
    full_dataset[response_col] - full_dataset[response_col].mean()
) / full_dataset[
    response_col
].std()  # centered + std
standardized_full_dataset: pd.DataFrame = (
    full_dataset - full_dataset.mean()
) / full_dataset.std()

standardized_ds: pd.DataFrame = standardized_features.copy()
standardized_ds["Balance"] = standardized_response.copy()


params: np.array = np.array([10**i for i in range(-2, 7)])
alphas: np.array = np.array([i / 5 for i in range(6)])

print(standardized_ds)

with mlflow.start_run():
    for alpha in alphas:
        weights: list = []
        for param in params:
            enr: ElasticNet = ElasticNet(
                alpha=alpha,
                l1_ratio=param,
                max_iter=1000000,
                normalize=False,
                fit_intercept=False,
            )
            enr.fit(standardized_features, standardized_response)
            weights.append(enr.coef_)
        weights = np.asmatrix(weights)
        i = 0
        l2_yaxis: list = [
            (np.array(weights[:, i].flatten()).flatten())
            for i in range(len(np.array(weights[i]).flatten()))
        ]
        for l in l2_yaxis:
            for i, ll in enumerate(l):
                # print(ll)
                name: str = standardized_ds.columns[i]
                mlflow.log_metric(name, ll)
