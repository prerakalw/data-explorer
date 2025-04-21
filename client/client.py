import asyncio
import json
import sys
import argparse
import subprocess
import os
from typing import Dict, Any, List, Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
import websockets
import jsonrpc

console = Console()

class MCPClient:
    def __init__(self, server_path: str, env_vars: Dict[str, str]):
        self.server_path = server_path
        self.env_vars = env_vars
        self.process = None
        self.websocket = None
        self.session = PromptSession(history=InMemoryHistory())
        self.request_id = 0
        self.command_completer = None

    async def start_server(self):
        """Start the MCP server process"""
        try:
            # Start the server process
            env = os.environ.copy()
            env.update(self.env_vars)

            self.process = subprocess.Popen(
                ["python", self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )

            # Give the server time to start
            await asyncio.sleep(1)

            console.print(f"[green]MCP server started (PID: {self.process.pid})[/green]")

        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            raise

    async def connect(self):
        """Connect to the MCP server via WebSocket"""
        try:
            # Connect to WebSocket
            self.websocket = await websockets.connect("ws://localhost:8765")
            console.print("[green]Connected to MCP server[/green]")

            # Initialize session
            await self.send_request("initialize", {
                "clientInfo": {
                    "name": "mcp-cli",
                    "version": "0.1.0"
                },
                "protocolVersion": "2024-11-05"
            })

            # Get capabilities
            response = await self.send_request("initialized", {})

            # List available commands and create completer
            tools = await self.send_request("tools/list", {})
            prompts = await self.send_request("prompts/list", {})
            resources = await self.send_request("resources/list", {})

            commands = (
                    ["list tools", "list prompts", "list resources", "execute", "prompt", "resource", "help", "exit"] +
                    [f"execute {tool['name']}" for tool in tools.get('tools', [])] +
                    [f"prompt {prompt['name']}" for prompt in prompts.get('prompts', [])] +
                    [f"resource {resource['name']}" for resource in resources.get('resources', [])]
            )

            self.command_completer = WordCompleter(commands, ignore_case=True)

        except Exception as e:
            console.print(f"[red]Error connecting to server: {e}[/red]")
            raise

    def get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to server"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method,
            "params": params
        }

        await self.websocket.send(json.dumps(request))
        response_str = await self.websocket.recv()
        response = json.loads(response_str)

        if "error" in response:
            raise Exception(f"Server error: {response['error']}")

        return response.get("result", {})

    async def list_resources(self):
        """List available resources"""
        resources = await self.send_request("resources/list", {})

        table = Table(title="Available Resources")
        table.add_column("Name", style="cyan")
        table.add_column("URI", style="green")
        table.add_column("Description", style="yellow")

        for resource in resources.get("resources", []):
            table.add_row(
                resource.get("name", ""),
                resource.get("uri", ""),
                resource.get("description", "")
            )

        console.print(table)

    async def list_tools(self):
        """List available tools"""
        tools = await self.send_request("tools/list", {})

        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="yellow")

        for tool in tools.get("tools", []):
            table.add_row(
                tool.get("name", ""),
                tool.get("description", "")
            )

        console.print(table)

    async def list_prompts(self):
        """List available prompts"""
        prompts = await self.send_request("prompts/list", {})

        table = Table(title="Available Prompts")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="yellow")

        for prompt in prompts.get("prompts", []):
            table.add_row(
                prompt.get("name", ""),
                prompt.get("description", "")
            )

        console.print(table)

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]):
        """Execute a tool"""
        console.print(f"[yellow]Executing tool: {tool_name}[/yellow]")

        result = await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": args
        })

        contents = result.get("content", [])
        for content in contents:
            if content.get("type") == "text":
                syntax = Syntax(content.get("text", ""), "json", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"Tool Result: {tool_name}"))
            elif content.get("type") == "image":
                console.print("[blue]Image generated. Base64 data available in the response.[/blue]")

    async def get_prompt(self, prompt_name: str, args: Dict[str, Any]):
        """Get a prompt"""
        console.print(f"[yellow]Getting prompt: {prompt_name}[/yellow]")

        result = await self.send_request("prompts/get", {
            "name": prompt_name,
            "arguments": args
        })

        console.print(f"[green]Description:[/green] {result.get('description', '')}")
        console.print("\n[green]Messages:[/green]")

        for message in result.get("messages", []):
            role = message.get("role", "")
            content = message.get("content", {})

            if content.get("type") == "text":
                markdown = Markdown(content.get("text", ""))
                console.print(Panel(markdown, title=f"Role: {role}"))

    async def read_resource(self, uri: str):
        """Read a resource"""
        console.print(f"[yellow]Reading resource: {uri}[/yellow]")

        result = await self.send_request("resources/read", {
            "uri": uri
        })

        contents = result.get("contents", [])
        for content in contents:
            if content.get("type") == "text":
                try:
                    # Try to parse as JSON for better formatting
                    data = json.loads(content.get("text", "{}"))
                    syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=f"Resource: {uri}"))
                except:
                    # Fallback to plain text
                    console.print(Panel(content.get("text", ""), title=f"Resource: {uri}"))

    def parse_command(self, command: str) -> tuple[str, str, Dict[str, Any]]:
        """Parse user command into action, target, and arguments"""
        parts = command.strip().split(maxsplit=2)

        if len(parts) == 0:
            return "", "", {}

        action = parts[0].lower()
        target = parts[1] if len(parts) > 1 else ""
        args = {}

        # Parse JSON arguments if provided
        if len(parts) > 2:
            try:
                args = json.loads(parts[2])
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON arguments[/red]")

        return action, target, args

    async def handle_command(self, command: str):
        """Handle a user command"""
        if not command.strip():
            return

        action, target, args = self.parse_command(command)

        try:
            if action == "help":
                self.show_help()
            elif action == "exit":
                return False
            elif action == "list":
                if target == "resources":
                    await self.list_resources()
                elif target == "tools":
                    await self.list_tools()
                elif target == "prompts":
                    await self.list_prompts()
                else:
                    console.print(f"[red]Unknown list command: {target}[/red]")
            elif action == "execute":
                if not target:
                    console.print("[red]Please specify a tool name[/red]")
                elif not args:
                    console.print("[yellow]No arguments provided. Enter JSON arguments:[/yellow]")
                    args_str = await self.session.prompt_async("Arguments (JSON): ")
                    args = json.loads(args_str)

                await self.execute_tool(target, args)
            elif action == "prompt":
                if not target:
                    console.print("[red]Please specify a prompt name[/red]")
                elif not args:
                    console.print("[yellow]No arguments provided. Enter JSON arguments:[/yellow]")
                    args_str = await self.session.prompt_async("Arguments (JSON): ")
                    args = json.loads(args_str)

                await self.get_prompt(target, args)
            elif action == "resource":
                if not target:
                    console.print("[red]Please specify a resource URI[/red]")
                await self.read_resource(target)
            else:
                console.print(f"[red]Unknown command: {action}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        return True

    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  help                      Show this help message
  exit                      Exit the client
  list resources            List available resources
  list tools                List available tools
  list prompts             List available prompts
  execute <tool> [json]    Execute a tool with optional JSON arguments
  prompt <name> [json]     Get a prompt with optional JSON arguments
  resource <uri>           Read a resource by URI

Examples:
  list tools
  execute execute_query {"query": "SELECT * FROM APP_CONFIG WHERE ROWNUM <= 10"}
  execute analyze_configurations {"table_name": "APP_CONFIG", "tenant_column": "COMPANY_ID", "config_columns": ["SETTING_A", "SETTING_B"]}
  prompt config_similarity {"table_name": "APP_CONFIG"}
  resource oracle://config/app_config
        """
        console.print(Panel(help_text, title="MCP Client Help"))

    async def run(self):
        """Run the interactive client"""
        try:
            await self.start_server()
            await self.connect()

            console.print("[green]MCP Client Ready. Type 'help' for commands.[/green]")

            while True:
                try:
                    command = await self.session.prompt_async(
                        "mcp> ",
                        completer=self.command_completer
                    )

                    if not await self.handle_command(command):
                        break
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        if self.websocket:
            await self.websocket.close()

        if self.process:
            self.process.terminate()
            self.process.wait()
            console.print("[yellow]MCP server stopped[/yellow]")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCP Command Line Client")
    parser.add_argument("--server", required=True, help="Path to MCP server script")
    parser.add_argument("--host", default="localhost", help="Oracle host")
    parser.add_argument("--port", type=int, default=1521, help="Oracle port")
    parser.add_argument("--service-name", default="ORCL", help="Oracle service name")
    parser.add_argument("--user", required=True, help="Oracle username")
    parser.add_argument("--password", required=True, help="Oracle password")

    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()

    env_vars = {
        "ORACLE_HOST": args.host,
        "ORACLE_PORT": str(args.port),
        "ORACLE_SERVICE_NAME": args.service_name,
        "ORACLE_USER": args.user,
        "ORACLE_PASSWORD": args.password
    }

    client = MCPClient(args.server, env_vars)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())