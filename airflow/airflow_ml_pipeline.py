from airflow import DAG
import pandas as pd
import datetime as dt
from datetime import timedelta
import os
import numpy as np

from elasticsearch import Elasticsearch, helpers
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.empty import EmptyOperator
# untuk connect postgre dengan python
import psycopg2 as db

# Model
import pickle
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Extract data from postgresql
def extract_data():
    conn_string = (
    "dbname='final_project' "
    "host='postgres' "
    "user='airflow' "
    "password='airflow' "
    "port=5432"
)
    conn=db.connect(conn_string)

    # query from sql
    final_project_dataset = pd.read_sql("select * from final_project", conn)

    # Saving raw data to CSV
    final_project_dataset.to_csv('/opt/airflow/dags/dataset_raw.csv', index=False)
    print("-------Students Saved------")

# Data Cleaning Process
def transforms():
    # Data Loading
    df = pd.read_csv('/opt/airflow/dags/dataset_raw.csv')

    # Drop Duplicate Data
    df = df.drop_duplicates()

    # Imputasi kolom kategorikal dengan modus
    categorical_cols = ["Weather", "Traffic_Level", "Time_of_Day"]
    for col in categorical_cols:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

    # Imputasi kolom numerik dengan mean
    numeric_cols = ["Courier_Experience_yrs"]
    for col in numeric_cols:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
    
    # Lowercase columns name
    df.columns = df.columns.str.lower()

    # Saving Clean Data to CSV
    df.to_csv('/opt/airflow/dags/dataset_clean.csv', index=False)
    print("-------Data Saved------")

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [c.strip().lower() for c in X.columns]  # pastikan konsisten lowercase

        # Pastikan kolom wajib ada
        required = {"distance_km", "delivery_time_min", "preparation_time_min", "courier_experience_yrs"}
        if required.issubset(X.columns):

            # Hindari pembagian nol
            X["delivery_time_min"] = X["delivery_time_min"].replace(0, np.nan).fillna(X["delivery_time_min"].median())

            # Fitur baru
            X["speed_km_per_min"] = X["distance_km"] / X["delivery_time_min"]
            X["prep_to_deliv_ratio"] = X["preparation_time_min"] / X["delivery_time_min"]

            # Tangani inf/NaN
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X["speed_km_per_min"] = X["speed_km_per_min"].fillna(0)
            X["prep_to_deliv_ratio"] = X["prep_to_deliv_ratio"].fillna(0)

            # Bikin experience_level
            X["courier_experience_yrs"] = X["courier_experience_yrs"].fillna(0).clip(lower=0, upper=20)
            X["experience_level"] = pd.cut(
                X["courier_experience_yrs"],
                bins=[0, 2, 5, 10, 20],
                labels=["Newbie", "Intermediate", "Experienced", "Veteran"],
                include_lowest=True
            ).astype(object).fillna("Newbie")

        else:
            # Jika dataset tidak punya kolom wajib, buat default
            X["speed_km_per_min"] = 0
            X["prep_to_deliv_ratio"] = 0
            X["experience_level"] = "Newbie"

        return X

def model():
    # Data Loading
    df = pd.read_csv("/opt/airflow/dags/dataset_clean.csv")

    # --- 1. Feature Engineering ---
    df["speed_km_per_min"] = df["distance_km"] / df["delivery_time_min"]
    df["prep_to_deliv_ratio"] = df["preparation_time_min"] / df["delivery_time_min"]
    df["experience_level"] = pd.cut(
    df["courier_experience_yrs"],
    bins=[0, 2, 5, 10, 20],
    labels=["Newbie", "Intermediate", "Experienced", "Veteran"])

    # --- 2. Pisahkan fitur dan target ---
    X = df.drop(columns=["delivery_time_min", "order_id"])
    y = df["delivery_time_min"]

    num_features = ['distance_km', 'preparation_time_min', 'courier_experience_yrs',
                'speed_km_per_min', 'prep_to_deliv_ratio']

    cat_features = ['weather', 'traffic_level', 'experience_level']

    # --- 4. Preprocessing pipeline ---
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # --- 5. Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    best_xgb_model = Pipeline(steps=[
        ("feature_engineering", FeatureEngineering()),
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42
        ))
    ])

    best_xgb_model.fit(X_train, y_train)
    os.makedirs("/opt/airflow/dags/models", exist_ok=True)
    with open("/opt/airflow/dags/models/model.pkl", "wb") as f:
        pickle.dump(best_xgb_model, f)

def evaluate_model(**context):
    df = pd.read_csv("/opt/airflow/dags/dataset_clean.csv")

    X = df.drop("delivery_time_min", axis=1)
    y = df["delivery_time_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = "/opt/airflow/dags/models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"Model Evaluation Results:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

    # Simpan hasil evaluasi ke file log
    os.makedirs("/opt/airflow/dags/logs", exist_ok=True)
    with open("/opt/airflow/dags/logs/evaluation_log.txt", "a") as log:
        log.write(f"{dt.datetime.now()} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}\n")

def deploy_model(**context):
    source = "/opt/airflow/dags/models/model.pkl"
    destination = "/opt/airflow/dags/deployment/best_model.pkl"

    os.makedirs("/opt/airflow/dags/deployment", exist_ok=True)
    os.replace(source, destination)
    print(f"Model deployed to {destination}")

default_args = {
    'owner': 'food_delivery_ml_pipeline',
    'start_date': dt.datetime(2025, 11, 1, 14) - dt.timedelta(hours = 7),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('Final_Project_Postgres_ETL',
         default_args=default_args,
         schedule_interval= '* * 1 * *',     
         catchup= False) as dag:

    print_starting = BashOperator(task_id='starting',
                               bash_command='echo "I am reading the CSV now....."')
    
    extractData = PythonOperator(task_id='Extract_Data_From_Postgresql',
                             python_callable=extract_data)
    
    transformData = PythonOperator(task_id='Data_Cleaning',
                                   python_callable=transforms)
    
    modeling = PythonOperator(task_id='Modeling',
                                   python_callable=model)
    
    evaluateModel = PythonOperator(task_id='Model_Evaluation',
                                   python_callable=evaluate_model)
    
    deployModel = PythonOperator(task_id='Model_Saving',
                                   python_callable=deploy_model)
    
    print_stop = BashOperator(task_id='stopping',
                               bash_command='echo "I done converting the CSV"')


print_starting >> extractData >> transformData >> modeling >> evaluateModel >> deployModel >> print_stop


