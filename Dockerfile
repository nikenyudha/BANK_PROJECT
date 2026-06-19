FROM python:3.11-slim

WORKDIR /code

# Install library OS yang dibutuhkan oleh LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# EXPOSE untuk mendokumentasikan port GCP
EXPOSE 8080
EXPOSE 7860

# Hugging Face Spaces berjalan di port 7860 secara default
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

# HANYA GUNAKAN SATU CMD 
# Perintah shell ini pintar: Kalau GCP kasih variabel $PORT (8080), dia pakai 8080. 
# Kalau GCP ga ngasih apa-apa (seperti di Hugging Face), dia otomatis fallback ke 7860.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]