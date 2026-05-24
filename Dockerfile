# Multi-Agent Supervisor System — Docker 镜像
# 构建: docker build -t multi-agent-supervisor .
# 运行: docker run -p 8003:8003 --env-file .env multi-agent-supervisor

FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (SQLite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi uvicorn \
    langchain langchain-community langchain-core \
    langchain-openai langgraph langgraph-checkpoint \
    langgraph-prebuilt langgraph-supervisor \
    sqlalchemy python-dotenv

# 复制项目代码
COPY main.py api.py ./
COPY .env.example ./

# 创建数据目录
RUN mkdir -p /app/logs

# 暴露 FastAPI 端口
EXPOSE 8003

# 启动 FastAPI 服务
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8003"]
