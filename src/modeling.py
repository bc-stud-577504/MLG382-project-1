
import pandas as pd
import numpy as np

from category_encoders import OneHotEncoder
import skimpy as sk
import pytimetk as tk

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib as jl

# Supress warnings
import warnings
warnings.simplefilter(action="ignore", category=Warning)

# Load data
df = pd.read_csv("./data/prepared_data.csv")

# Prepare data for model
target = "Loan_Status"

x = df.drop(
    columns=[target],
    inplace=False
)
y = df[target]

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the model
model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)

# Train the model
model_lr.fit(x_train, y_train)

jl.dump(model_lr, "./artifacts/model_lr.sav")

y_pred = model_lr.predict(x_test)
y_pred_df = pd.DataFrame(y_pred, columns=["Predictions"])

# Save predictions
y_pred_df.to_csv("./artifacts/predictions.csv", index=False)