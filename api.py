"""Multi-Agent Supervisor System — FastAPI 接口"""
import os, uuid, json, time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

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
    """同步调用 — 等待完整回复"""
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


@app.post("/ask/stream")
async def ask_agent_stream(q: Question):
    """
    流式调用 — Server-Sent Events (SSE)

    客户端可以实时看到 Agent 的每一步决策:
      - supervisor 节点 → 路由决策
      - music_agent 节点 → LLM 思考 + Tool 调用 + 最终回答
      - invoice_agent 节点 → 同上

    SSE 事件格式:
      data: {"event": "node_start", "node": "supervisor", "timestamp": "..."}
      data: {"event": "llm_token", "content": "AC/DC", "node": "music_agent"}
      data: {"event": "tool_call", "tool": "get_albums_by_artist", "args": {...}}
      data: {"event": "node_end", "node": "music_agent", "duration_ms": 1234}
      data: [DONE]
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = {
        "messages": [HumanMessage(content=q.question)],
        "customer_id": q.customer_id,
        "loaded_memory": "None",
        "route": "",
    }

    async def event_generator():
        node_start = time.time()
        final_answer = ""

        for chunk in agent.stream(state, config, subgraphs=True):
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                continue

            _, node_output = chunk
            if not isinstance(node_output, dict):
                continue

            # subgraphs=True 时, dict 的 key 是节点名, value 是该节点的输出
            for node_name, node_data in node_output.items():
                if not isinstance(node_data, dict):
                    continue

                yield f"data: {json.dumps({'event': 'node_start', 'node': str(node_name)}, ensure_ascii=False)}\n\n"

                # 路由决策
                route = node_data.get("route", "")
                if route:
                    yield f"data: {json.dumps({'event': 'router_decision', 'route': route}, ensure_ascii=False)}\n\n"

                # 消息内容
                msgs = node_data.get("messages", [])
                for msg in msgs:
                    tc = getattr(msg, "tool_calls", None)
                    if tc:
                        for t in tc:
                            yield f"data: {json.dumps({'event': 'tool_call', 'tool': t.get('name', '?'), 'args': str(t.get('args', {}))[:200]}, ensure_ascii=False)}\n\n"
                    content = getattr(msg, "content", "")
                    if content:
                        final_answer = str(content)
                        yield f"data: {json.dumps({'event': 'llm_output', 'content': content[:500]}, ensure_ascii=False)}\n\n"

        total_ms = round((time.time() - node_start) * 1000, 1)
        yield f"data: {json.dumps({'event': 'complete', 'duration_ms': total_ms, 'answer': final_answer[:500]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "multi-agent-supervisor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
