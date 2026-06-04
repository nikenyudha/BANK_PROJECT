FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Hugging Face Spaces berjalan di port 7860 secara default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]