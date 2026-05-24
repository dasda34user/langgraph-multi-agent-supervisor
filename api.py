"""Multi-Agent Supervisor System — FastAPI 接口"""
import os, uuid
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# 复用 main.py 中的 agent 和 db 初始化
from main import agent, db

app = FastAPI(title="Multi-Agent Supervisor API", version="1.0")


class Question(BaseModel):
    question: str = Field(description="用户问题")
    customer_id: str = Field(default="1", description="客户 ID")


class Response(BaseModel):
    question: str
    customer_id: str
    answer: str


@app.post("/ask", response_model=Response)
def ask_agent(q: Question):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = {
        "messages": [HumanMessage(content=q.question)],
        "customer_id": q.customer_id,
        "loaded_memory": "None",
        "route": "",
    }
    result = agent.invoke(state, config)
    answer = result["messages"][-1].content
    return Response(question=q.question, customer_id=q.customer_id, answer=answer)


@app.get("/health")
def health():
    return {"status": "ok", "service": "multi-agent-supervisor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
