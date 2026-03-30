FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8502

# 创建启动脚本
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'python main.py &' >> /app/start.sh && \
    echo 'sleep 10' >> /app/start.sh && \
    echo 'streamlit run app.py --server.port 8502 --server.address 0.0.0.0' >> /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]
