FROM apache/airflow:2.7.3-python3.9

# Gunakan root hanya untuk install dependencies sistem
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        make \
        libc-dev \
        libgomp1 \
        libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Kembali ke user airflow untuk install python packages
USER airflow

# Jalankan pip sebagai user airflow (bukan root)
RUN pip install --no-cache-dir \
    "xgboost==2.0.3" \
    "scikit-learn==1.3.2" \
    "pandas==2.0.3" \
    "numpy==1.26.4" \
    "matplotlib==3.8.2" \
    "seaborn==0.13.2"

