# 面试材料包 — Supervisor 多 Agent 系统

## Profile

- 目标岗位：AI 智能体开发实习生（AI Agent Development Intern）
- 当前水平：完成过双工具 ReAct Agent 和三路路由 RAG Agent，本项目为高级进阶
- 技术栈：Python + LangGraph 1.x (StateGraph) + DeepSeek API + SQLAlchemy + SQLite
- 时间预算：已投入 1 天
- 资源：Windows 11，无 GPU/Docker，DeepSeek API
- 运行深度：smoke-test + 关键架构改造（langgraph_supervisor → 手动 StateGraph）
- 项目来源：[FareedKhan-dev/Multi-Agent-AI-System](https://github.com/FareedKhan-dev/Multi-Agent-AI-System)

---

## STAR 简历项目（4-5 行）

**Supervisor 多 Agent 智能系统 — 基于 LangGraph StateGraph 手动构建** — 个人项目

- 针对 AI Agent 岗位对多智能体架构的设计要求，手动构建 LangGraph StateGraph 实现 Supervisor 模式：LLM 分类器 → 条件边 → 两个专用子 Agent 的路由架构，替换原项目的黑盒 `langgraph_supervisor` 为可控的显式条件路由
- 设计 Music Catalog Sub-Agent（4 个 SQL 工具：专辑/歌曲/流派/模糊搜索）和 Invoice Sub-Agent（3 个 SQL 工具：发票/消费/员工查询），每个子 Agent 采用双轮 LLM 调用模式（① LLM 决策调用 Tool → ② LLM 基于 Tool 结果生成回答）
- 集成 Chinook 数据库（SQLite 内存数据库，11 张表，100+ 条真实数据），使用 LangChain SQLDatabase + SQLAlchemy 实现 Agent 对结构化数据的动态查询
- 产出：三场景验证通过（音乐查询：2 张专辑 / 发票查询：7 条记录 / 歌曲搜索：命中判定）；用 FastAPI + Pydantic 封装为 RESTful 服务，通过 `uv run python api.py` 启动 HTTP 接口；项目也可 `uv run python main.py` 直接运行

---

## 面试官拷问 Q&A

### Q1: 什么是 Supervisor 多 Agent 模式？和你之前做的单 Agent 有什么本质区别？

**单 Agent**（如 agents 项目）：一个 LLM 绑几个 Tool，收到问题 → 想一想要不要调工具 → 调用 → 回答。

**Supervisor 多 Agent**：一个调度者 LLM + 多个专业子 Agent。每个子 Agent 有自己的 Tool 集和自己的决策循环。

```
单 Agent:                     Supervisor 多 Agent:
User → [LLM + Tools] → Ans    User → [Supervisor]
                                          │
                               ┌──────────┴──────────┐
                               ▼                      ▼
                          [Music Agent]         [Invoice Agent]
                          (音乐Tool×4)           (发票Tool×3)
                               │                      │
                               └──────────┬──────────┘
                                          ▼
                                        Ans
```

**本质区别**：
1. **关注点分离**：Music Agent 只知道音乐表结构，Invoice Agent 只管发票表。各自 Prompt 更精准
2. **独立扩展**：加一个 "Customer Service Agent" 只需加一个节点 + 一条条件边
3. **失败隔离**：Music Agent 挂了不影响 Invoice Agent

### Q2: 为什么不用 `langgraph_supervisor.create_supervisor` 而要手动构建 StateGraph？

我试过 `create_supervisor`。它的原理是给 Supervisor LLM 绑定 `transfer_to_xxx` 工具，期望 LLM 自动调用转移工具。但 DeepSeek 在工具调用上的行为与 OpenAI 有差异——部分情况下 Supervisor 不调用转移工具直接回复，或者 Transfer 后子 Agent 收到的消息上下文被污染。

**手动方案的优势**：
1. 分类和路由是显式的 `add_conditional_edges`，不是 LLM 自由发挥
2. 子 Agent 收到的是干净的 `HumanMessage`，不被 Supervisor 的转移消息干扰
3. 两轮 LLM 调用模式（①决策调Tool → ②综合Tool结果回答）是手写的，每一步都可控可调

面试时我会说："`create_supervisor` 适合快速原型，但生产环境我更倾向手动 StateGraph——路由逻辑确定性更高，调试也容易。"

### Q3: 子 Agent 的双轮 LLM 调用模式是怎么设计的？

```
第 1 轮: LLM (带 Tool) → 决定调哪个 Tool → 执行 Tool
第 2 轮: LLM (不带 Tool) → 基于 Tool 结果 → 生成自然语言回答
```

**代码对应**：
```python
# 第 1 轮：LLM 决策 + 工具执行
response = llm_music.invoke([SystemMessage(prompt), user_msg])
if response.tool_calls:
    for tc in response.tool_calls:
        tool_result = find_tool(tc["name"]).invoke(tc["args"])

# 第 2 轮：LLM 综合生成
final_response = llm.invoke(
    f"Based on tool results:\n{tool_results}\nProvide answer:"
)
```

**为什么分两轮而不是一轮？**
- 第一轮的 LLM 带 Tool binding，输出是结构化的 tool_call
- 第二轮的 LLM 不带 Tool，专注于"把 SQL 结果翻译成用户看得懂的话"
- 分开后每轮职责单一，Prompt 更精准，Token 浪费更少

### Q4: Supervisor 分类器是怎么设计的？

```python
def supervisor_node(state, config):
    prompt = f"""Route the user's question:
    - "music" for: artists, albums, tracks, genres, songs
    - "invoice" for: invoices, purchases, billing, payments
    User question: {question}
    Reply with ONLY one word: music or invoice"""

    response = llm.invoke(prompt)
    return {"route": "invoice" if "invoice" in response.content else "music"}
```

**设计要点**：
1. **纯分类，不绑 Tool**：分类器不需要 `bind_tools()`，只需要理解意图
2. **输出约束为单个词**："Reply with ONLY one word"——减少 LLM 自由发挥
3. **兜底逻辑**：`"invoice" in response.content` 做模糊匹配，即使 LLM 多输出几个字也能正确解析
4. **不是正则匹配**：不能用 `if "发票" in question` 关键词匹配，因为用户可能说 "我上次买了什么？"——需要语义理解

### Q5: 如果我要给系统加第三个 Agent（如 "Employee Agent"），怎么做？

只需四步：

```python
# 1. 定义新 Tool
@tool
def get_employee_info(employee_id: str) -> str:
    return db.run(f"SELECT * FROM Employee WHERE EmployeeId = {employee_id};")

# 2. 创建新 Agent 节点
def employee_node(state, config): ...

# 3. 修改条件路由
def route_decision(state):
    if route == "invoice": return "invoice_agent"
    if route == "employee": return "employee_agent"
    return "music_agent"

# 4. 加到图里
wf.add_node("employee_agent", employee_node)
wf.add_conditional_edges("supervisor", route_decision, {
    "music_agent": "music_agent",
    "invoice_agent": "invoice_agent",
    "employee_agent": "employee_agent",  # 新增
})
```

**这就是 StateGraph 的模块化优势**：加一个 Agent 不影响已有 Agent。

### Q6: 为什么用 SQLite 内存数据库而不是持久化数据库？

面试要诚实回答：
- **演示/学习场景**：SQLite `:memory:` 模式启动快、零配置、退出即销毁
- **生产场景应改为**：持久化 SQLite 文件或 PostgreSQL，并在每次修改后重新加载
- 当前方案展示的是 **Agent 与结构化数据交互的能力**，数据库选型是工程细节，不是核心技术壁垒

### Q7: 如果 DeepSeek 返回了错误的 SQL 或者 Tool 调用失败，你怎么处理？

当前代码在每个 Tool 函数里做了基础防护（try/except 返回错误信息），但完整方案应该是：

1. **Tool 执行层**：`try/except` 捕获异常，返回有意义错误给 LLM
2. **LLM 重试层**：如果 Tool 返回错误，让 LLM 看到错误信息后重试（最多 2-3 次）
3. **兜底层**：连续失败后返回 "抱歉，我暂时无法查询到相关信息，请稍后再试"

面试时可以提到 LangGraph 的 `Command` 机制支持动态重路由——Tool 失败后不返回 END，而是回到 Agent 节点再试一次。

### Q8: 三个项目的技术演进路线是什么？为什么这样安排学习路径？

```
agents (入门)           OLGA (中级)            multi-agent (高级)
═══════════════        ════════════════       ═══════════════════
1 Agent + 2 Tools      3 路路由 + RAG         Supervisor 多 Agent
ReAct 循环             FAISS 向量检索         手动 StateGraph
@tool 装饰器           Persona + Memory       SQL + 条件路由
FastAPI 部署           路由分类器              双轮 LLM 调用
```

**演进逻辑**：
1. agents → 学会 Tool 是什么、Agent 怎么用 Tool
2. OLGA → 学会什么时候不该用 Agent（路由分类）、怎么给 Agent 加外部知识（RAG）
3. multi-agent → 学会多个 Agent 怎么协作、怎么手动设计图结构

面试官问 "你做过的最复杂的 Agent 项目是什么" → 讲 multi-agent 的 Supervisor 手动 StateGraph。问 "你对 RAG 的理解" → 讲 OLGA 的 FAISS 嵌入链路。问 "Agent 基础" → 讲 agents 的 ReAct 循环。

---

## 核心代码讲解稿（面试现场 5 分钟）

### 启动

```bash
cd D:\FILE\CODE\py\multi-agent
uv run python main.py
```

### 架构图

```
START
  │
  ▼
supervisor (LLM 分类: music / invoice)
  │
  ├── route="music" → music_agent
  │     ├─ 1st Round: LLM + Tools → SQL query
  │     └─ 2nd Round: LLM → natural answer
  │
  └── route="invoice" → invoice_agent
        ├─ 1st Round: LLM + Tools → SQL query
        └─ 2nd Round: LLM → natural answer
  │
  ▼
 END
```

### 关键代码片段

**StateGraph 组装（4 行核心）**：
```python
wf = StateGraph(MultiAgentState)
wf.add_node("supervisor", supervisor_node)
wf.add_node("music_agent", music_node)
wf.add_node("invoice_agent", invoice_node)
wf.add_edge(START, "supervisor")
wf.add_conditional_edges("supervisor", route_decision,
    {"music_agent": "music_agent", "invoice_agent": "invoice_agent"})
wf.add_edge("music_agent", END)
wf.add_edge("invoice_agent", END)
```

**条件路由函数**：
```python
def route_decision(state: MultiAgentState):
    return "invoice_agent" if state.get("route") == "invoice" else "music_agent"
```

**7 个 SQL Tool**：全部用 `@tool` 装饰器,底层是 LangChain `SQLDatabase.run()`,直接执行 SQL。

### 我的改动（与原 notebook 对比）

| 变更 | 原因 |
|------|------|
| `langgraph_supervisor.create_supervisor` → 手动 StateGraph | DeepSeek 工具调用兼容性，且手动方案可控性更高 |
| 子 Agent 双轮 LLM 调用 | 手动处理 tool call → tool result → final answer 流程 |
| 子 Agent 只接收最后一条 HumanMessage | 避免 Supervisor 转移消息污染上下文 |
| LLM 统一为 DeepSeek | 原项目用 Nebius + Together，改为单 API |
| `langgraph_supervisor` + `langgraph-swarm` + 25 个依赖 → 精简 | 去掉不必要的包，加快安装 |
| Jupyter Notebook → 单文件 Python 脚本 | 可直接运行，方便演示 |

---

## PPT 提示词

> 帮我做一个技术面试的 PPT，主题是 "Supervisor 多 Agent 智能系统"。包含：
>
> 1. **封面**：项目名称 "Multi-Agent Supervisor System" + 岗位 "AI Agent 开发实习生" + 技术栈 (LangGraph / StateGraph / SQLite / DeepSeek)
> 2. **为什么多 Agent**：一页对比单 Agent vs Supervisor 多 Agent，突出关注点分离、独立扩展、失败隔离
> 3. **架构图（核心页）**：START → Supervisor (LLM 分类) → [Music Agent / Invoice Agent] → END。每个子 Agent 展开为 双轮 LLM 调用
> 4. **Supervisor 分类器设计**：纯 LLM 分类（不绑 Tool）、输出约束、兜底逻辑、为什么不用关键词匹配
> 5. **子 Agent 双轮调用**：第一轮 LLM+Tool 决策，第二轮 LLM 自然语言生成。展示代码片段
> 6. **手动 StateGraph vs create_supervisor**：一张表对比两个方案，标注我的选择和原因
> 7. **验证结果**：3 个场景的输入/路由/工具调用/输出，每个一条线
> 8. **三个项目演进路线**：agents → OLGA → multi-agent，一句话定位每小题。展示技术广度
> 9. **限制与规划**：加第三个 Agent 只需 4 步、生产环境改造方向
>
> 风格：深色底色，蓝色主色，每页 3-5 个要点，架构图用流程图风格。

---

## 投递检查表

- [ ] 简历中 multi-agent 项目是否命中 JD 关键词：智能体设计、LangChain/LangGraph、多 Agent 协作、Python？
- [ ] 是否能用口语讲清 "Supervisor 分类 → 条件路由 → 子 Agent 双轮调用 → 结果" 全链路？
- [ ] 是否准备好回答 "为什么手动 StateGraph 而不是 create_supervisor"（DeepSeek 兼容性 + 可控性）？
- [ ] 是否准备好回答 "怎么加第三个 Agent"（4 步法）？
- [ ] 是否准备好回答 "三个项目的技术演进路线"（展示系统学习规划）？
- [ ] 是否准备好回答 "子 Agent 的双轮 LLM 调用模式"？
- [ ] 项目是否可以随时 `uv run python main.py` 演示给面试官？
- [ ] `.env` 是否在 `.gitignore` 中？

---

## 三个项目面试策略总表

| | agents | OLGA | multi-agent |
|---|---|---|---|
| **定位** | 入门：Tool-using | 中级：RAG + 路由 | 高级：多 Agent 协作 |
| **主推场景** | 被追问"还做过什么" | 被追问"懂 RAG 吗" | **面试主项目** |
| **JD 命中** | Tool 调用、ReAct | FAISS、向量检索 | StateGraph、多 Agent |
| **一句话** | Agent 怎么用工具 | Agent 怎么有记忆 | 多个 Agent 怎么协作 |

**面试策略**：简历放 multi-agent 作为主项目，面试时先讲它（10 分钟深度），被追问其他经验时自然引出 OLGA（RAG）和 agents（基础），形成 "从入门到高级的系统学习路径" 印象。最后说 "下一步想把三个项目的优势合并——既有 RAG 知识库，又有 Tool 调用，还有多 Agent 协作" 展示规划力。
