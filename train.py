import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
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


tuning_params: np.array = np.array([10**i for i in range(-2, 7)])

print(standardized_ds)

with mlflow.start_run():
    weights: list = []
    for alpha in tuning_params:
        model: Ridge = Ridge(
            alpha=alpha,
            max_iter=1000000,
            fit_intercept=False,
        )
        model.fit(standardized_features, standardized_response)
        # print(model.coef_)
        weights.append(model.coef_)
        weight_metrics: dict = {f"feature_{k}": v for k, v in enumerate(model.coef_)}
        mlflow.log_metrics(weight_metrics)
    weights = np.asmatrix(weights)
    # i = 0
    # l2_yaxis: list = [
    #     np.array(weights[:, i].flatten()).flatten()
    #     for i in range(len(np.array(weights[i]).flatten()))
    # ]
    # regression_coeff_l2: np.array = np.transpose(l2_yaxis)
