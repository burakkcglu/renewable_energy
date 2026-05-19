FROM python:3.11-slim

# Sistem bağımlılıklarını yükle (Optimizasyon matrisleri için build araçları)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Kütüphaneleri önbellekten (cache) hızlıca yüklemek için önce requirements kopyalanır
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Notebook portunu dışarıya aç
EXPOSE 8888

# Token ve şifre zorunluluğunu kaldırarak jupyter'i otomatik başlatıyoruz
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]