"""
数据库抽象层 — SQLite (开发) / PostgreSQL (生产) 双模支持

用法:
    from db import get_database
    db = get_database()          # 根据 DB_URL 环境变量自动选择
    db = get_database("sqlite")  # 强制 SQLite
    db = get_database("postgres") # 强制 PostgreSQL

SQLite 开发模式:
    DB_URL=sqlite:///chinook.db     (文件持久化)
    DB_URL=sqlite:///:memory:       (内存模式, 默认)

PostgreSQL 生产模式:
    DB_URL=postgresql://user:pass@host:5432/chinook

SQLite WAL 模式:
    默认开启 WAL (Write-Ahead Logging), 提升并发读性能。
    SQLite 默认 journal_mode=delete → WAL 支持多读一写, 适合 Agent 的查询密集型场景。
"""

import os, sys, sqlite3, urllib.request
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

CHINOOK_URL = ("https://raw.githubusercontent.com/lerocha/chinook-database/"
               "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql")


def _setup_sqlite(db_url: str):
    """初始化 SQLite (内存或文件) 并导入 Chinook 数据"""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    sql_path = Path("Chinook_Sqlite.sql")
    if not sql_path.exists():
        print("Downloading Chinook database...", file=sys.stderr)
        urllib.request.urlretrieve(CHINOOK_URL, sql_path)

    if ":memory:" in db_url:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(sql_path.read_text(encoding="utf-8"))
        engine = create_engine(
            "sqlite://", creator=lambda: connection,
            poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    else:
        # 文件模式: 先创建并导入数据
        db_file = db_url.replace("sqlite:///", "")
        if not Path(db_file).exists():
            connection = sqlite3.connect(db_file)
            connection.executescript(sql_path.read_text(encoding="utf-8"))
            connection.close()

        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            pool_size=5,           # 连接池大小
            max_overflow=10,        # 最大溢出连接
            pool_pre_ping=True,     # 每次使用前检查连接是否有效
        )

    # 开启 WAL 模式 (提升并发读性能)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        conn.exec_driver_sql("PRAGMA cache_size=-64000;")  # 64MB 缓存
        conn.commit()

    return engine


def _setup_postgres(db_url: str):
    """初始化 PostgreSQL 连接"""
    from sqlalchemy import create_engine

    engine = create_engine(
        db_url,
        pool_size=10,              # PG 支持更大连接池
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,         # 每小时回收连接
    )

    # PostgreSQL 需要手动导入 Chinook 数据
    sql_path = Path("Chinook_Sqlite.sql")
    if not sql_path.exists():
        print("Downloading Chinook SQL...")
        urllib.request.urlretrieve(CHINOOK_URL, sql_path)

    print("PostgreSQL: manual data import required. Run:")
    print(f"  psql {db_url} < Chinook_Sqlite.sql")

    return engine


def get_database(backend: str = None):
    """
    根据配置返回数据库引擎。

    优先级: 参数 backend > 环境变量 DB_BACKEND > 默认 sqlite
    """
    backend = backend or os.getenv("DB_BACKEND", "sqlite")

    if backend == "postgres":
        db_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/chinook")
        return _setup_postgres(db_url)
    else:
        db_url = os.getenv("DATABASE_URL", "sqlite:///:memory:")
        return _setup_sqlite(db_url)


# ═══════════════════════════════════════════════════════════
# SQLite vs PostgreSQL — 关键差异 (面试必问)
# ═══════════════════════════════════════════════════════════
#
# | 维度           | SQLite                    | PostgreSQL               |
# |---------------|--------------------------|--------------------------|
# | 并发模型       | 单写多读 (WAL模式)        | MVCC 多版本并发控制       |
# | 连接池         | 不推荐 >10 连接           | 支持数百连接              |
# | 数据类型       | 弱类型 (动态)              | 强类型 (严格)             |
# | 网络访问       | 嵌入式, 不支持远程         | TCP/IP 远程连接           |
# | 事务隔离       | Serializable (串行化)      | Read Committed (默认)     |
# | 适用场景       | 开发/单机/嵌入式            | 生产/多实例/高并发         |
# | 全文搜索       | FTS5 扩展                  | pg_trgm + GIN 索引        |
#
# WAL 模式原理:
#   传统 SQLite (journal_mode=delete):
#     写操作 → 先写回滚日志 → 修改数据库 → 删除日志
#     写操作期间阻塞所有读操作
#
#   WAL 模式 (journal_mode=WAL):
#     写操作 → 追加到 WAL 文件 (不修改主数据库)
#     读操作 → 读主数据库 (不被写阻塞)
#     定期 checkpoint: 将 WAL 合并回主数据库
#
#   结果: 读和写可以同时进行, 适合 Agent 的"LLM 等待期间其他查询可继续"场景
