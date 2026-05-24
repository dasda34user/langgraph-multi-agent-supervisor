"""
MCP (Model Context Protocol) Server — 标准化 Agent-Tool 通信

将 multi-agent 的 7 个 SQL Tool 暴露为 MCP 标准接口。
任何支持 MCP 的 Agent 框架都可以直接调用这些工具。

MCP 规范: https://modelcontextprotocol.io
协议: JSON-RPC 2.0 over stdio

用法:
    uv run python mcp/server.py
    # 通过 stdio 与 MCP Client 通信
"""

import sys, json, uuid
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import music_tools, invoice_tools, db

# 所有可用工具
ALL_TOOLS = music_tools + invoice_tools


def _tool_to_mcp_schema(tool) -> dict:
    """将 LangChain Tool 转换为 MCP Tool Schema"""
    # LangChain tool 的 args_schema 包含参数定义
    schema = tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
    }


def handle_request(request: dict) -> dict:
    """处理单个 JSON-RPC 请求"""
    method = request.get("method", "")
    req_id = request.get("id", 0)
    params = request.get("params", {})

    # tools/list — 列出所有可用工具
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [_tool_to_mcp_schema(t) for t in ALL_TOOLS]
            }
        }

    # tools/call — 调用指定工具
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        for tool in ALL_TOOLS:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(arguments)
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": str(result)}]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": f"Tool execution failed: {str(e)}"}
                    }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
        }

    # initialize — MCP 握手
    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "multi-agent-mcp", "version": "1.0.0"},
                "capabilities": {"tools": {}}
            }
        }

    # notifications/initialized — 无需响应
    elif method == "notifications/initialized":
        return None

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


def main():
    """MCP Server 主循环 — stdio JSON-RPC"""
    print(f"MCP Server starting with {len(ALL_TOOLS)} tools...", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            if response:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            print(json.dumps(error_response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
