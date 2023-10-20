import sklearn
print(sklearn.__version__)

import pickle

from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor

app = Flask(__name__)


@app.route("/")
def index():
    cat_cols = {
        'IsFirstTimeHomebuyer': ['n', 'y'],
        'Occupancy': ['o', 'i', 's'],
        'Channel': ['t', 'r', 'c', 'b'],
        'PPM': ['n', 'y'],
        'PropertyState': ['il', 'co', 'ks', 'ca', 'nj', 'wi', 'fl', 'ct', 'ga', 'tx', 'md',
                          'ma', 'sc', 'wy', 'nc', 'az', 'in', 'ms', 'ny', 'wa', 'ar', 'va',
                          'mn', 'la', 'pa', 'or', 'ri', 'ut', 'mi', 'tn', 'al', 'mo', 'ia',
                          'nm', 'nv', 'oh', 'ne', 'vt', 'hi', 'id', 'pr', 'dc', 'gu', 'ky',
                          'nh', 'sd', 'me', 'mt', 'ok', 'wv', 'de', 'nd', 'ak'],
        'PropertyType': ['sf', 'pu', 'co', 'mh', 'cp', 'lh'],
        'LoanPurpose': ['p', 'n', 'c']
    }

    num_cols = {
        'FirstPayment_Month': 0,
        'OrigUPB': 117000,
        'OrigInterestRate': 6.75,
        'OrigLoanTerm': 360,
        'MonthsDelinquent': 0,
        'Credit_range': 1,
        'LTV_range': 3,
        'Repay_range': 2,
        'NumBorrowers': 2,
        'Maturity_Month': 10,
        'MIP': 2,
        'DTI': 0,
    }

    return render_template("index.html", cat_cols=cat_cols, num_cols=num_cols)


@app.route("/", methods=['POST'])
def predict():
    prediction = ""
    message = ""
    if request.method == 'POST':
        XGBreg_pipeline = pickle.load(open('XGBreg_pipeline.pkl', 'rb'))

        data = {}
        for variable in [   'FirstPayment_Month', 'IsFirstTimeHomebuyer', 'Maturity_Month', 'MIP', 'Occupancy',
                            'DTI', 'OrigUPB', 'OrigInterestRate', 'Channel', 'PPM',
                            'PropertyState', 'PropertyType', 'LoanPurpose',
                            'OrigLoanTerm', 'NumBorrowers', 'MonthsDelinquent',
                            'Credit_range', 'LTV_range', 'Repay_range'
                            ]:
            data[variable] = request.form.get(variable)

        sample = pd.DataFrame([data])

        message = "Prepayment Risk Ratio Prediction..."
        prediction = XGBreg_pipeline.predict(sample)[0] * 100
    return render_template('results.html', prediction=prediction, message=message)



@app.route("/tr", methods=['POST'])
def train():
    data = pd.read_csv("datasets/X_full.csv")

    X = data
    y = data["PPR"]

    cat_cols = data.select_dtypes("object").columns
    num_cols = data.select_dtypes(exclude="object").columns

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # pipelines categorical variables
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant")),
            ("one_hot_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # pipeline of num variables
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean"))
        ]
    )

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    # now let's build our main pipeline called (bai pipeline)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=15)),
            ("model", XGBRegressor()),
        ]
    )

    message = ""
    if request.method == 'POST':
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        pickle.dump(pipeline, open("XGBreg_pipeline.pkl", "wb"))

        message = f"Model Accuracy : {accuracy}"

    return render_template('results.html', prediction="", message=message)



if __name__ == '__main__':
    app.run(debug=True)
