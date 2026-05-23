"""
Multi-Agent AI System — 手动构建 Supervisor StateGraph
原项目: FareedKhan-dev/Multi-Agent-AI-System
适配: DeepSeek API, 手动条件路由（不用 langgraph_supervisor）

架构:
  __start__
      │
  supervisor (LLM 分类: music / invoice)
      │
      ├── music_catalog_subagent (ReAct + 4 SQL tools)
      │       └── → END
      └── invoice_subagent (ReAct + 3 SQL tools)
              └── → END
"""

import os, sqlite3, urllib.request, uuid
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# ═══════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════

llm = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)

checkpointer = MemorySaver()
in_memory_store = InMemoryStore()

# ═══════════════════════════════════════════════════════════
# 2. Chinook 数据库
# ═══════════════════════════════════════════════════════════

CHINOOK_URL = ("https://raw.githubusercontent.com/lerocha/chinook-database/"
               "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql")

def get_chinook_db():
    from langchain_community.utilities import SQLDatabase
    sql_path = Path("Chinook_Sqlite.sql")
    if not sql_path.exists():
        print("Downloading Chinook database...")
        urllib.request.urlretrieve(CHINOOK_URL, sql_path)
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_path.read_text(encoding="utf-8"))
    engine = create_engine(
        "sqlite://", creator=lambda: connection,
        poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    return SQLDatabase(engine=engine)

print("Setting up Chinook database...")
db = get_chinook_db()

# ═══════════════════════════════════════════════════════════
# 3. State — 多 Agent 共享状态
# ═══════════════════════════════════════════════════════════

class MultiAgentState(TypedDict):
    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps
    route: str  # "music" / "invoice" / "done"

# ═══════════════════════════════════════════════════════════
# 4. Music Catalog Tools
# ═══════════════════════════════════════════════════════════

@tool
def get_albums_by_artist(artist: str) -> str:
    """按艺人名查找专辑列表."""
    r = db.run(
        f"SELECT Album.Title FROM Album "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"WHERE Artist.Name = '{artist}';"
    )
    return r if r.strip() else f"No albums for '{artist}'."

@tool
def get_tracks_by_artist(artist: str) -> str:
    """按艺人名查找歌曲列表."""
    r = db.run(
        f"SELECT Track.Name FROM Track "
        f"JOIN Album ON Track.AlbumId = Album.AlbumId "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"WHERE Artist.Name = '{artist}' LIMIT 20;"
    )
    return r if r.strip() else f"No tracks for '{artist}'."

@tool
def get_songs_by_genre(genre: str) -> str:
    """按音乐流派查找歌曲."""
    r = db.run(
        f"SELECT Track.Name, Artist.Name FROM Track "
        f"JOIN Album ON Track.AlbumId = Album.AlbumId "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"JOIN Genre ON Track.GenreId = Genre.GenreId "
        f"WHERE Genre.Name = '{genre}' LIMIT 10;"
    )
    return r if r.strip() else f"No tracks in genre '{genre}'."

@tool
def check_for_songs(song_title: str) -> str:
    """模糊搜索某首歌是否存在."""
    r = db.run(
        f"SELECT Track.Name, Artist.Name, Album.Title FROM Track "
        f"JOIN Album ON Track.AlbumId = Album.AlbumId "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"WHERE Track.Name LIKE '%{song_title}%' LIMIT 10;"
    )
    return r if r.strip() else f"No songs matching '{song_title}'."

music_tools = [get_albums_by_artist, get_tracks_by_artist,
               get_songs_by_genre, check_for_songs]
llm_music = llm.bind_tools(music_tools)

# ═══════════════════════════════════════════════════════════
# 5. Invoice Tools
# ═══════════════════════════════════════════════════════════

@tool
def get_invoices_by_date(customer_id: str) -> str:
    """查询某客户的发票，按日期倒序。"""
    return db.run(
        f"SELECT InvoiceId, InvoiceDate, Total FROM Invoice "
        f"WHERE CustomerId = {customer_id} "
        f"ORDER BY InvoiceDate DESC LIMIT 20;"
    )

@tool
def get_invoices_by_price(customer_id: str) -> str:
    """查询某客户最贵的购买记录。"""
    return db.run(
        f"SELECT InvoiceLine.UnitPrice, Track.Name FROM InvoiceLine "
        f"JOIN Track ON InvoiceLine.TrackId = Track.TrackId "
        f"JOIN Invoice ON InvoiceLine.InvoiceId = Invoice.InvoiceId "
        f"WHERE Invoice.CustomerId = {customer_id} "
        f"ORDER BY InvoiceLine.UnitPrice DESC LIMIT 10;"
    )

@tool
def get_employee_by_invoice(invoice_id: str, customer_id: str) -> str:
    """查询处理某发票的员工信息。"""
    return db.run(
        f"SELECT Employee.FirstName, Employee.LastName, Employee.Title "
        f"FROM Employee JOIN Customer ON Employee.EmployeeId = Customer.SupportRepId "
        f"JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId "
        f"WHERE Invoice.InvoiceId = {invoice_id} "
        f"AND Invoice.CustomerId = {customer_id};"
    )

invoice_tools = [get_invoices_by_date, get_invoices_by_price,
                 get_employee_by_invoice]
llm_invoice = llm.bind_tools(invoice_tools)

# ═══════════════════════════════════════════════════════════
# 6. Supervisor 节点 — LLM 分类路由
# ═══════════════════════════════════════════════════════════

classifier = llm.bind_tools([])  # 不绑工具，纯分类

def supervisor_node(state: MultiAgentState, config):
    """Supervisor: 分析用户意图，决定路由到哪个子 Agent"""
    user_msg = state["messages"][-1]
    question = user_msg.content if hasattr(user_msg, 'content') else str(user_msg)

    prompt = f"""You are a router for a digital music store's customer support system.
Analyze the user's question and decide which department should handle it.

- Return "music" for questions about: artists, albums, tracks, genres, songs, music catalog
- Return "invoice" for questions about: invoices, purchases, billing, payments, employee, orders

User question: {question}

Reply with ONLY one word: music or invoice"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    if "invoice" in route:
        decision = "invoice"
    else:
        decision = "music"

    return {"route": decision}

# ═══════════════════════════════════════════════════════════
# 7. Music Sub-Agent 节点
# ═══════════════════════════════════════════════════════════

def music_node(state: MultiAgentState, config):
    """Music Agent: 使用 SQL 工具回答音乐相关问题"""
    cid = state.get("customer_id", "")
    sys_prompt = f"""You are a Music Catalog Assistant for a digital music store.
Current customer ID: {cid}
Use SQL tools to look up information about artists, albums, tracks, and genres.
Always call the appropriate tool to get real database results — do NOT make up data."""

    # 只传入用户原始消息，不传 supervisor 消息
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    response = llm_music.invoke(
        [SystemMessage(sys_prompt)] + user_msgs[-1:]  # 只给最后一条用户消息
    )

    # 如果是 tool call，执行工具并获取结果
    if response.tool_calls:
        tool_results = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            for t in music_tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results.append(f"Tool: {tool_name}\nArgs: {tool_args}\nResult: {result}")
                    break

        # 第二轮：让 LLM 基于工具结果生成最终回答
        final_prompt = f"""Based on the tool results below, provide a helpful answer to the user's question.
User question: {user_msgs[-1].content if user_msgs else 'N/A'}

Tool results:
{chr(10).join(tool_results)}

Now provide the final answer:"""

        final_response = llm.invoke(final_prompt)
        return {"messages": [AIMessage(content=final_response.content)]}

    return {"messages": [response]}

# ═══════════════════════════════════════════════════════════
# 8. Invoice Sub-Agent 节点
# ═══════════════════════════════════════════════════════════

def invoice_node(state: MultiAgentState, config):
    """Invoice Agent: 使用 SQL 工具回答发票相关问题"""
    cid = state.get("customer_id", "1")
    sys_prompt = f"""You are an Invoice & Billing Assistant for a digital music store.
The customer ID is: {cid}
Use SQL tools to look up this customer's invoices, purchases, and billing history.
Always call the appropriate tool with customer_id={cid} — do NOT make up data."""

    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    response = llm_invoice.invoke(
        [SystemMessage(sys_prompt)] + user_msgs[-1:]
    )

    if response.tool_calls:
        tool_results = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            for t in invoice_tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results.append(f"Tool: {tool_name}\nArgs: {tool_args}\nResult: {result}")
                    break

        final_prompt = f"""Based on the tool results below, provide a helpful answer to the user's question.
User question: {user_msgs[-1].content if user_msgs else 'N/A'}

Tool results:
{chr(10).join(tool_results)}

Now provide the final answer:"""

        final_response = llm.invoke(final_prompt)
        return {"messages": [AIMessage(content=final_response.content)]}

    return {"messages": [response]}

# ═══════════════════════════════════════════════════════════
# 9. 条件路由函数
# ═══════════════════════════════════════════════════════════

def route_decision(state: MultiAgentState) -> Literal["music_agent", "invoice_agent"]:
    """根据 Supervisor 的分类结果路由"""
    if state.get("route") == "invoice":
        return "invoice_agent"
    return "music_agent"

# ═══════════════════════════════════════════════════════════
# 10. 组装 StateGraph
# ═══════════════════════════════════════════════════════════

wf = StateGraph(MultiAgentState)

wf.add_node("supervisor", supervisor_node)
wf.add_node("music_agent", music_node)
wf.add_node("invoice_agent", invoice_node)

wf.add_edge(START, "supervisor")
wf.add_conditional_edges(
    "supervisor", route_decision,
    {"music_agent": "music_agent", "invoice_agent": "invoice_agent"}
)
wf.add_edge("music_agent", END)
wf.add_edge("invoice_agent", END)

agent = wf.compile(
    name="multi_agent_supervisor",
    checkpointer=checkpointer,
    store=in_memory_store
)

# ═══════════════════════════════════════════════════════════
# 11. Test
# ═══════════════════════════════════════════════════════════

def ask(question: str, customer_id: str = "1"):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = {
        "messages": [HumanMessage(content=question)],
        "customer_id": customer_id,
        "loaded_memory": "None",
        "route": ""
    }
    result = agent.invoke(state, config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Supervisor — Manual StateGraph")
    print("=" * 60)

    # Test 1: Music
    print("\n[Test 1: Music] What albums does AC/DC have?")
    print(f"Route: ", end="")
    ans = ask("What albums does the artist 'AC/DC' have?")
    print(ans[:500])

    # Test 2: Invoice
    print("\n[Test 2: Invoice] Show my recent invoices.")
    print(f"Route: ", end="")
    ans = ask("Show me my recent invoices.", customer_id="1")
    print(ans[:500])

    # Test 3: complex — check song
    print("\n[Test 3: Music Search] Do you have the song 'Thunderstruck'?")
    ans = ask("Do you have the song 'Thunderstruck' in your catalog?")
    print(ans[:500])
