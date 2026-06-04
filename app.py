from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import numpy as np
import json
import sqlite3
import os
from datetime import datetime

# Mengunci path database secara absolut agar sama dengan drift_detector.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "production_logs.db")

app = FastAPI(
    title="Bank Marketing Predictor API (Hugging Face Edition)",
    description="API level Senior dengan Request Logging otomatis ke SQLite.",
    version="1.1.0"
)

# Konfigurasi Hugging Face
HF_REPO_ID = "nikenlarash22/bank-marketing-model"

# Variabel global untuk menampung model dan skema kolom di memori
model = None
model_columns = None

# --- FUNGSIONALITAS DATABASE (REQUEST LOGGING) ---
def init_db():
    """Membuat tabel database log jika belum ada"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            duration INTEGER,
            campaign INTEGER,
            pdays INTEGER,
            previous INTEGER,
            euribor3m REAL,
            prediction INTEGER,
            probability_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(age, duration, campaign, pdays, previous, euribor3m, prediction, probability):
    """Menyimpan data input penting dan output model ke database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_logs 
        (timestamp, age, duration, campaign, pdays, previous, euribor3m, prediction, probability_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        age, duration, campaign, pdays, previous, euribor3m, prediction, probability
    ))
    conn.commit()
    conn.close()

# --- LOAD MODEL & INIT DB SAAT API STARTUP ---
@app.on_event("startup")
def startup_event():
    global model, model_columns
    
    # 1. Jalankan pembuatan database dan tabel
    init_db()
    print("🗄️ Database Logs Berhasil Diinisialisasi di:", DB_FILE)
    
    # 2. Download model dari Hugging Face Hub
    try:
        print("📥 Sedang mengunduh model dari Hugging Face Hub...")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="bank_deposit_model_v1.pkl")
        columns_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model_columns.json")
        
        model = joblib.load(model_path)
        with open(columns_path, 'r') as f:
            model_columns = json.load(f)
            
        print("✅ Model dan Skema Kolom berhasil dimuat ke memori!")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")

# --- DEFINE API CONTRACT USING PYDANTIC ---
class ClientDataInput(BaseModel):
    age: int = Field(..., example=35)
    duration: int = Field(..., example=250)
    campaign: int = Field(..., example=1)
    pdays: int = Field(..., example=999)
    previous: int = Field(..., example=0)
    emp_var_rate: float = Field(..., alias="emp.var.rate", example=1.1)
    cons_price_idx: float = Field(..., alias="cons.price.idx", example=93.994)
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", example=-36.4)
    euribor3m: float = Field(..., example=4.857)
    nr_employed: float = Field(..., alias="nr.employed", example=5191.0)
    month: str = Field(..., example="may")
    day_of_week: str = Field(..., example="mon")
    job: str = Field(..., example="admin.")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="university.degree")
    default: str = Field(..., example="no")
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    poutcome: str = Field(..., example="nonexistent")

    class Config:
        populate_by_name = True

# --- ENDPOINT PREDIKSI ---
@app.post("/predict")
def predict_deposit(client: ClientDataInput):
    if model is None or model_columns is None:
        raise HTTPException(status_code=503, detail="Model belum siap di memori server.")
        
    try:
        input_data = client.model_dump(by_alias=False)
        
        # --- FEATURE ENGINEERING ---
        month_map = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
        
        m_num = month_map.get(input_data['month'].lower(), 5)
        d_num = day_map.get(input_data['day_of_week'].lower(), 1)
        
        input_data['month_sin'] = np.sin(2 * np.pi * m_num / 12)
        input_data['month_cos'] = np.cos(2 * np.pi * m_num / 12)
        input_data['day_sin'] = np.sin(2 * np.pi * d_num / 5)
        input_data['day_cos'] = np.cos(2 * np.pi * d_num / 5)
        input_data['economic_stress_index'] = input_data['emp_var_rate'] * input_data['cons_price_idx']
        
        df_input = pd.DataFrame([input_data])
        df_features = pd.DataFrame(0, index=[0], columns=model_columns)
        
        for col in model_columns:
            if col in df_input.columns:
                df_features[col] = df_input[col]
        
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        for cat in categorical_features:
            val = df_input.loc[0, cat]
            col_name = f"{cat}_{val}"
            if col_name in df_features.columns:
                df_features[col_name] = 1

        # INFERENCE
        prediction = model.predict(df_features)[0]
        probability = model.predict_proba(df_features)[0][1]
        
        # 📥 CATAT KE DATABASE (Sistem Request Logging Aktif!)
        log_to_db(
            age=input_data['age'],
            duration=input_data['duration'],
            campaign=input_data['campaign'],
            pdays=input_data['pdays'],
            previous=input_data['previous'],
            euribor3m=input_data['euribor3m'],
            prediction=int(prediction),
            probability=float(probability)
        )
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "prediction_label": "yes" if prediction == 1 else "no",
            "probability_score": float(probability)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT UNTUK CEK LOGS ---
@app.get("/logs", summary="Cek Histori Prediksi di Database")
def get_logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        df_logs = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df_logs.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Bank Marketing API via Hugging Face is online with Logging!"}