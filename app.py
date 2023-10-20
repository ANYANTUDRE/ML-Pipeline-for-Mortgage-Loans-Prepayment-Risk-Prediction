from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import numpy as np

import sklearn
print("Current version of scikit-learn", sklearn.__version__)


app = Flask(__name__)

cat_cols = {
        'PropertyState': ['il', 'co', 'ks', 'ca', 'nj', 'wi', 'fl', 'ct', 'ga', 'tx', 'md',
                          'ma', 'sc', 'wy', 'nc', 'az', 'in', 'ms', 'ny', 'wa', 'ar', 'va',
                          'mn', 'la', 'pa', 'or', 'ri', 'ut', 'mi', 'tn', 'al', 'mo', 'ia',
                          'nm', 'nv', 'oh', 'ne', 'vt', 'hi', 'id', 'pr', 'dc', 'gu', 'ky',
                          'nh', 'sd', 'me', 'mt', 'ok', 'wv', 'de', 'nd', 'ak'],
        'PropertyType': ['sf', 'pu', 'co', 'mh', 'cp', 'lh'],
        'Occupancy': ['o', 'i', 's'],
        'IsFirstTimeHomebuyer': ['n', 'y'],
        'Channel': ['t', 'r', 'c', 'b'],
        'PPM': ['n', 'y'],
        'LoanPurpose': ['p', 'n', 'c'],
    }

num_cols = {
        'MonthsDelinquent': 0,
        'Maturity_Month': 1,
        'Credit_range': 1,
        'LTV_range': 1,
        'Repay_range': 1,
        'FirstPayment_Month': 1,
        'OrigUPB': 100000,
        'OrigInterestRate': 1,
        'OrigLoanTerm': 100,
        'NumBorrowers': 1,
        'MIP': 1,
        'DTI': 0,
    }

@app.route("/")
def index():
    return render_template("index.html", cat_cols=cat_cols, num_cols=num_cols)


@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        XGBreg_pipeline = pickle.load(open('XGBreg_pipeline.pkl', 'rb'))

        data = {}
        for variable in [   'Credit_range', 'LTV_range', 'Repay_range', 'FirstPayment_Month', 'IsFirstTimeHomebuyer', 
                            'Maturity_Month', 'MIP', 'Occupancy', 'DTI', 'OrigUPB', 'OrigInterestRate', 'Channel', 
                            'PPM', 'PropertyState', 'PropertyType', 'LoanPurpose', 'OrigLoanTerm', 'NumBorrowers', 'MonthsDelinquent'
                            ]:
            data[variable] = request.form.get(variable)

        sample = pd.DataFrame([data])

        #print("Before labeling", sample.info())
        # Label encoding for categoricals
        for colname in cat_cols:
            sample[colname], _ = sample[colname].factorize()
        #print("After labeling", sample.info())

        prediction = XGBreg_pipeline.predict(sample)[0]
    return render_template('results.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
