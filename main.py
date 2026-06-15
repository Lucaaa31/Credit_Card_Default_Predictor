import pandas as pd
import numpy as np
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


TARGET = "default.payment.next.month"
DEV_PATH = "dev.csv"
EVAL_PATH = "eval.csv"
OUT_PATH = "submission.csv"


BEST_PARAMS = dict(
    n_estimators = 650,
    max_depth = 35,
    min_samples_split = 9,
    min_samples_leaf = 7,
    max_features = "sqrt",
    bootstrap = True,
    class_weight = "balanced",
    random_state = 42,
    n_jobs = -1,
)

BEST_THRESHOLD = 0.5

PAY_COLS  = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
BILL_COLS = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
PAMT_COLS = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
FEAT_COLS = None 



def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(X_arr: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(X_arr, columns=FEAT_COLS)


    df["UTIL_RATE"] = df["BILL_AMT1"] / (df["LIMIT_BAL"].abs() + 1)
    for i, (p, b) in enumerate(zip(PAMT_COLS, BILL_COLS), 1):
        df[f"PAY_RATIO{i}"] = df[p] / (df[b].abs() + 1)
    df["MAX_DELAY"] = df[PAY_COLS].max(axis=1)
    df["AVG_DELAY"] = df[PAY_COLS].mean(axis=1)
    df["N_DELAYED"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["DELAY_TREND"] = df["PAY_0"] - df["PAY_6"]
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["TOTAL_BILL"] = df[BILL_COLS].sum(axis=1)
    df["TOTAL_PAY"] = df[PAMT_COLS].sum(axis=1)
    df["AVG_PAY_RATIO"] = df["TOTAL_PAY"] / (df["TOTAL_BILL"].abs() + 1)

    out = df.values.astype(float)
    out = np.where(np.isinf(out), 0.0, out)
    out = np.where(np.isnan(out), 0.0, out)
    return out



def main():
    global FEAT_COLS


    dev = clean(pd.read_csv(DEV_PATH))
    eval_df = clean(pd.read_csv(EVAL_PATH))

    FEAT_COLS = [c for c in dev.columns if c not in ["ID", TARGET]]

    X_train = dev[FEAT_COLS].values
    y_train = dev[TARGET].values
    X_eval = eval_df[FEAT_COLS].values

    print(f"Train : {X_train.shape[0]:,} samples  |  "
          f"Default rate: {y_train.mean()*100:.1f}%")
    print(f"Eval  : {X_eval.shape[0]:,} samples")


    fe = FunctionTransformer(add_features, validate=False)
    pipe = Pipeline([
        ("fe", fe),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**BEST_PARAMS)),
    ])

    print("Training the model...")
    pipe.fit(X_train, y_train)

    print("Predicting the eval set...")
    proba  = pipe.predict_proba(X_eval)[:, 1]
    y_pred = (proba >= BEST_THRESHOLD).astype(int)

    submission = pd.DataFrame({"Id": eval_df["ID"].values, "Predicted": y_pred})
    submission.to_csv(OUT_PATH, index=False)

    print(f"Submission saved on {OUT_PATH}")

if __name__ == "__main__":
    main()
