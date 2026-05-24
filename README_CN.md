# Multi-Agent Supervisor 系统 — 学习指南

## 项目概述

这是一个**手动构建的 LangGraph StateGraph 多 Agent 系统**。与使用 `langgraph_supervisor.create_supervisor()` 的自动方案不同，本项目手动设计了每一个节点和每一条边。

## 核心概念

### 1. StateGraph 手动构建

```python
wf = StateGraph(MultiAgentState)        # 定义图
wf.add_node("supervisor", supervisor_node)   # 3 个节点
wf.add_node("music_agent", music_node)
wf.add_node("invoice_agent", invoice_node)
wf.add_edge(START, "supervisor")            # 入口边
wf.add_conditional_edges("supervisor", ...)  # 条件边
wf.add_edge("music_agent", END)             # 出口边
```

### 2. Supervisor 模式

Supervisor 不是另一个 Agent，而是一个 LLM 分类器：

```python
def supervisor_node(state, config):
    question = state["messages"][-1].content
    prompt = f"Route to music or invoice: {question}"
    response = llm.invoke(prompt)
    return {"route": "music" if "music" in response.content else "invoice"}
```

### 3. 子 Agent 双轮调用

```
第1轮: LLM + Tools → 决定调哪个 SQL 工具
第2轮: LLM (不带Tool) → 把 SQL 结果翻译成自然语言
```

### 4. 条件路由

```python
def route_decision(state):
    if state["route"] == "invoice":
        return "invoice_agent"
    return "music_agent"
```

## 文件结构

| 文件 | 作用 |
|------|------|
| `main.py` | Agent 定义 (7 个 SQL Tool + 3 个图节点 + StateGraph 组装) |
| `api.py` | FastAPI 接口 (同步 /ask + 流式 /ask/stream) |
| `db.py` | 数据库抽象层 (SQLite WAL / PostgreSQL 双模) |
| `Dockerfile` | 容器化构建 |
| `docker-compose.yml` | 一键部署 |

## 学习路径

1. **先看 StateGraph 组装** (`main.py` 最后 20 行) — 理解图的结构
2. **再看 supervisor_node** — 理解怎么分类路由
3. **再看 music_node** — 理解子 Agent 怎么调用 Tool
4. **最后看 route_decision** — 理解条件边怎么工作

## 启动

```bash
uv run python main.py              # 直接运行
uv run python api.py               # FastAPI (http://localhost:8003/docs)
docker compose up -d               # Docker 部署
```
