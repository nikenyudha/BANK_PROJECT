import sqlite3
import pandas as pd
import requests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "production_logs.db")

# Batas toleransi (Threshold) Data Drift berdasarkan karakteristik Dataset asli
# Jika data lapangan melompat melebihi batas ini, berarti tren pasar sudah berubah!
DRIFT_THRESHOLDS = {
    "max_acceptable_interest": 5.0,     # Batas atas suku bunga euribor3m
    "min_acceptable_duration": 150.0,   # Batas bawah rata-rata durasi telepon (dalam detik)
    "trigger_sample_count": 1000        # Setiap 1000 data baru masuk, cek potensi retraining
}

def check_data_drift():
    print("🔍 Menjalankan Pemeriksaan Data Drift berkala...")
    
    # 1. Tarik data log prediksi terbaru dari database SQLite
    try:
        conn = sqlite3.connect(DB_FILE)
        df_logs = pd.read_sql_query("SELECT * FROM prediction_logs", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Gagal membaca database: {e}")
        return

    total_logs = len(df_logs)
    print(f"📊 Total data nasabah yang sudah dilayani API saat ini: {total_logs} sampel.")

    if total_logs == 0:
        print("ℹ️ Belum ada data log yang tersimpan. Pemeriksaan dihentikan.")
        return

    # 2. Hitung statistik deskriptif dari data lapangan terbaru
    avg_duration = df_logs['duration'].mean()
    avg_interest = df_logs['euribor3m'].mean()

    print(f"--- ANALISIS TREN LAPANGAN ---")
    print(f"Rata-rata Durasi Telepon Saat Ini: {avg_duration:.2f} detik")
    print(f"Rata-rata Suku Bunga (Euribor3m) Saat Ini: {avg_interest:.4f}")

    # 3. ALGORITMA PENGAMBILAN KEPUTUSAN RETRAINING (RULES-BASED DRIFT)
    drift_detected = False
    reasons = []

    # Cek Kondisi 1: Apakah durasi telepon marketing menurun drastis?
    if avg_duration < DRIFT_THRESHOLDS["min_acceptable_duration"]:
        drift_detected = True
        reasons.append(f"Durasi telepon terlalu rendah ({avg_duration:.2f} detik). Tim Sales kurang optimal.")

    # Cek Kondisi 2: Apakah kondisi ekonomi makro bergeser ekstrem?
    if avg_interest > DRIFT_THRESHOLDS["max_acceptable_interest"]:
        drift_detected = True
        reasons.append(f"Suku bunga pasar melonjak terlalu tinggi ({avg_interest:.4f}).")

    # 4. EKSEKUSI REAKSI PIPELINE
    if drift_detected:
        print("\n🚨 [ALERT] DATA DRIFT TERDETEKSI! Karakteristik data lapangan sudah bergeser.")
        print(f"Alasan: {', '.join(reasons)}")
        print("🔄 Memulai Proses RETRAINING OTOMATIS... Mengirim instruksi ke pipeline training...")
        
        # Di level industri, di bagian ini menembak webhook (misal ke Airflow atau Jenkins)
        # untuk menjalankan ulang file notebook Colab  secara otomatis dari atas ke bawah.
        trigger_pipeline_retraining()
    else:
        print("\n✅ Aman! Kondisi data lapangan masih stabil dan sesuai dengan performa model saat ini.")

def trigger_pipeline_retraining():
    """Simulasi memicu retraining ulang model dengan data baru"""
    print("🚀 Sinyal dikirim! Mengambil data terbaru dari database bank, melatih ulang LightGBM, dan siap melakukan auto-push model v2 ke Hugging Face.")

if __name__ == "__main__":
    check_data_drift()