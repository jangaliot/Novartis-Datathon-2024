# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.
   This auxiliar file is intended to be used by participants in case
   you want to test the metric with your own train/validation splits."""

import pandas as pd
from pathlib import Path
from typing import Tuple
import pickle
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

def _CYME(df: pd.DataFrame) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""

    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"])/yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"])/df["target"]).median()

    return 1/2*(yearly_error + monthly_error)


def _metric(df: pd.DataFrame) -> float:
    """Compute metric of submission.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    # weight for each group
    zeros_weight = len(zeros)/len(df)
    recent_weight = 1 - zeros_weight

    # Compute CYME for each group
    return round(recent_weight*_CYME(recent) + zeros_weight*min(1,_CYME(zeros)),8)


def compute_metric(submission: pd.DataFrame) -> Tuple[float, float]:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """

    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[['cluster_nl', 'date', 'target', 'prediction', 'zero_actuals']]

    return _metric(submission)


# Load model
# model_path = "../models/weights/model_imputeKnn_imputeKnn_scale_xgboost_hpt.pkl"
# model = pickle.load(open(model_path, "rb"))

# Load data
train_data = pd.read_csv("../data/processed/train_data_processed_imputeKnn_scale.csv")
train_data_raw = pd.read_csv("../data/raw/train_data.csv")

# Inspección inicial
print("Dimensiones del dataset:", train_data.shape)
print(train_data.head())

# Identificar la variable objetivo y las características
target_col = "target"
features = [col for col in train_data.columns if (col != target_col and col != 'therapeutic_area')]

# Lista de categorías
categorias = [
    'THER_AREA_980E', 'THER_AREA_96D7', 'THER_AREA_6CEE', 'THER_AREA_644A',
    'THER_AREA_66C5', 'THER_AREA_CD59', 'THER_AREA_22ED', 'THER_AREA_645F',
    'THER_AREA_8E53', 'THER_AREA_4BA5', 'THER_AREA_051D', 'THER_AREA_032C'
]

# Diccionario para almacenar los modelos cargados
models_dict  = {}

# Cargar cada modelo y almacenarlo en el diccionario
for categoria in categorias:
    modelo_nombre = f"../models/weights/xgb_model_{categoria}.joblib"
    modelo_cargado = joblib.load(modelo_nombre)
    models_dict[categoria] = modelo_cargado

# Preprocesamiento: manejar valores categóricos y nulos
# Convertir categorías a valores numéricos
label_encoders = {}
for col in train_data.select_dtypes(include=['object']).columns:
    if col in features:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        label_encoders[col] = le

# Split into train and validation set
validation = train_data.sample(frac=0.2, random_state=42)

# Train your model
features = train_data.columns.difference(["target", "cluster_nl"])

# Lista para almacenar las predicciones
submission_predictions = []
list_the_areas = list(train_data_raw['therapeutic_area'])
# Iterar sobre cada fila del dataset de submission
for idx, row in validation.iterrows():
    # Identificar el área terapéutica de la fila
    therapeutic_area = list_the_areas[idx]
    
    # Seleccionar el modelo correspondiente
    if therapeutic_area in models_dict:
        model = models_dict[therapeutic_area]
        
        # Seleccionar las características relevantes para predicción
        row_features = row[features].values.reshape(1, -1)  # Convertir en un array 2D
        
        # Realizar la predicción
        prediction = model.predict(row_features)[0]  # Extraer el valor predicho
    else:
        # Asignar NaN si no hay modelo para el área terapéutica
        prediction = np.nan
        print(f"Advertencia: No se encontró modelo para el área terapéutica '{therapeutic_area}'. Se asigna NaN.")

    # Almacenar la predicción
    submission_predictions.append(prediction)

# Agregar las predicciones como una nueva columna en el DataFrame de submission
validation['prediction'] = submission_predictions

# Assign column ["zero_actuals"] in the depending if in your
# split the cluster_nl has already had actuals on train or not
train_clusters = train_data_raw["cluster_nl"].unique()
validation['cluster_nl'] = train_data_raw["cluster_nl"]
validation["zero_actuals"] = ~validation["cluster_nl"].isin(train_clusters)

# Optionally check performance
print("Performance:", compute_metric(validation))

# Prepare submission
submission_data = pd.read_csv("../data/processed/submission_data_processed_imputeKnn_scale.csv")
submission = pd.read_csv("../data/submissions/submission_template.csv")

categorical_columns = submission_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if col in label_encoders:  # Usa los label encoders existentes
        submission_data[col] = label_encoders[col].fit_transform(submission_data[col].astype(str))

# Fill in 'prediction' values of submission
submission["prediction"] = model.predict(submission_data[features])

# Save submission
ATTEMPT = "attempt_1"
submission.to_csv(f"../data/submissions/submission_{ATTEMPT}.csv", sep=",", index=False)
