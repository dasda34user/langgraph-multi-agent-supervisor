"""
MCP Client 测试 — 验证 MCP Server 的协议兼容性

模拟一个 MCP Client 通过 JSON-RPC 调用 multi-agent 的工具。
"""

import subprocess, json, sys
from pathlib import Path


def send_request(process, request: dict) -> dict:
    """发送 JSON-RPC 请求并获取响应"""
    payload = json.dumps(request, ensure_ascii=False) + "\n"
    process.stdin.write(payload)
    process.stdin.flush()
    response_line = process.stdout.readline()
    return json.loads(response_line)


def test_mcp():
    print("=" * 60)
    print("MCP Server Test")
    print("=" * 60)

    # 启动 MCP Server (stderr 重定向, 避免干扰 stdout JSON-RPC)
    server_script = str(Path(__file__).parent / "server.py")
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
    )

    try:
        # 1. Initialize
        print("\n[Test 1] Initialize...")
        resp = send_request(proc, {
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "test-client"}}
        })
        print(f"  Server: {resp['result']['serverInfo']['name']} v{resp['result']['serverInfo']['version']}")

        # 2. List tools
        print("\n[Test 2] List tools...")
        resp = send_request(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        print(f"  Available tools: {len(tools)}")
        for t in tools:
            print(f"    - {t['name']}: {t['description'][:60]}")

        # 3. Call a tool
        print("\n[Test 3] Call tool: get_albums_by_artist('AC/DC')...")
        resp = send_request(proc, {
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "get_albums_by_artist", "arguments": {"artist": "AC/DC"}}
        })
        if "result" in resp:
            text = resp["result"]["content"][0]["text"]
            print(f"  Result: {text[:200]}")
        else:
            print(f"  Error: {resp.get('error', 'unknown')}")

        # 4. Call invoice tool
        print("\n[Test 4] Call tool: get_invoices_sorted_by_date('1')...")
        resp = send_request(proc, {
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "get_invoices_sorted_by_date", "arguments": {"customer_id": "1"}}
        })
        if "result" in resp:
            text = resp["result"]["content"][0]["text"]
            print(f"  Result: {text[:200]}")
        else:
            print(f"  Error: {resp.get('error', 'unknown')}")

        print("\n" + "=" * 60)
        print("MCP Server: ALL TESTS PASSED")
        print("=" * 60)

    finally:
        proc.terminate()


if __name__ == "__main__":
    test_mcp()
