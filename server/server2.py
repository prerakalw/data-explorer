import logging
from typing import Dict, List, Any, Optional
import json
import os
import oracledb
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from dataclasses import dataclass
import websockets
from websockets.server import serve

# MCP specific imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OracleConfig:
    host: str
    port: int
    service_name: str
    user: str
    password: str

class ConfigAnalysisServer:
    def __init__(self, oracle_config: OracleConfig):
        self.server = Server("oracle-config-analysis")
        self.oracle_config = oracle_config
        self.pool = None
        self._register_handlers()

    async def initialize_pool(self):
        """Initialize Oracle connection pool"""
        if not self.pool:
            self.pool = oracledb.create_pool(
                user=self.oracle_config.user,
                password=self.oracle_config.password,
                host=self.oracle_config.host,
                port=self.oracle_config.port,
                service_name=self.oracle_config.service_name,
                min=2,
                max=10,
                increment=1
            )

    async def get_connection(self):
        """Get a connection from the pool"""
        if not self.pool:
            await self.initialize_pool()
        return self.pool.acquire()

    def _register_handlers(self):
        """Register all MCP protocol handlers"""

        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """List available configuration tables"""
            resources = []
            try:
                async with await self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT table_name, comments 
                            FROM user_tab_comments 
                            WHERE table_name LIKE '%CONFIG%'
                        """)
                        tables = cursor.fetchall()

                        for table_name, comments in tables:
                            resources.append(types.Resource(
                                uri=f"oracle://config/{table_name.lower()}",
                                name=table_name,
                                description=comments or f"Configuration table: {table_name}",
                                mimeType="application/json"
                            ))
            except Exception as e:
                logger.error(f"Error listing resources: {e}")

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read configuration table data"""
            if not uri.startswith("oracle://config/"):
                raise ValueError(f"Unknown resource URI: {uri}")

            table_name = uri.split("/")[-1].upper()

            try:
                async with await self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Get table metadata
                        cursor.execute("""
                            SELECT column_name, data_type 
                            FROM user_tab_columns 
                            WHERE table_name = :table_name
                        """, table_name=table_name)
                        columns = cursor.fetchall()

                        # Get row count
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]

                        # Get sample data (first 1000 rows)
                        cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= 1000")
                        rows = cursor.fetchall()

                        result = {
                            "table_name": table_name,
                            "metadata": {
                                "columns": [{"name": col[0], "type": col[1]} for col in columns],
                                "row_count": row_count
                            },
                            "sample_data": [
                                dict(zip([col[0] for col in columns], row))
                                for row in rows
                            ]
                        }

                        return json.dumps(result, default=str)
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                raise

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available analysis tools"""
            return [
                types.Tool(
                    name="execute_query",
                    description="Execute a read-only SQL query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to execute (SELECT only)"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="analyze_configurations",
                    description="Analyze configuration similarity across tenants",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Configuration table to analyze"},
                            "tenant_column": {"type": "string", "description": "Column that identifies the tenant/company"},
                            "config_columns": {"type": "array", "items": {"type": "string"}, "description": "Configuration columns to analyze"}
                        },
                        "required": ["table_name", "tenant_column", "config_columns"]
                    }
                ),
                types.Tool(
                    name="find_deviations",
                    description="Find configurations that deviate from common patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Configuration table to analyze"},
                            "config_column": {"type": "string", "description": "Configuration column to analyze"},
                            "threshold": {"type": "number", "description": "Percentage threshold for deviation detection (default: 20)"}
                        },
                        "required": ["table_name", "config_column"]
                    }
                ),
                types.Tool(
                    name="create_chart",
                    description="Create a chart from query results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to get chart data"},
                            "chart_type": {"type": "string", "enum": ["bar", "pie", "line", "scatter", "heatmap"], "description": "Type of chart"},
                            "x_column": {"type": "string", "description": "Column for x-axis"},
                            "y_column": {"type": "string", "description": "Column for y-axis"},
                            "title": {"type": "string", "description": "Chart title"},
                            "group_column": {"type": "string", "description": "Column for grouping (optional)"}
                        },
                        "required": ["query", "chart_type", "x_column", "y_column"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution requests"""
            try:
                if name == "execute_query":
                    return await self._execute_query(arguments["query"])
                elif name == "analyze_configurations":
                    return await self._analyze_configurations(
                        arguments["table_name"],
                        arguments["tenant_column"],
                        arguments["config_columns"]
                    )
                elif name == "find_deviations":
                    return await self._find_deviations(
                        arguments["table_name"],
                        arguments["config_column"],
                        arguments.get("threshold", 20)
                    )
                elif name == "create_chart":
                    return await self._create_chart(
                        arguments["query"],
                        arguments["chart_type"],
                        arguments["x_column"],
                        arguments["y_column"],
                        arguments.get("title", ""),
                        arguments.get("group_column")
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[types.Prompt]:
            """List available analysis prompts"""
            return [
                types.Prompt(
                    name="config_similarity",
                    description="Analyze configuration similarity across tenants",
                    arguments=[
                        types.PromptArgument(
                            name="table_name",
                            description="Name of the configuration table",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="config_outliers",
                    description="Find configuration outliers and deviations",
                    arguments=[
                        types.PromptArgument(
                            name="table_name",
                            description="Name of the configuration table",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="config_summary",
                    description="Generate a summary of configuration patterns",
                    arguments=[
                        types.PromptArgument(
                            name="table_name",
                            description="Name of the configuration table",
                            required=True
                        )
                    ]
                )
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
            """Generate analysis prompts"""
            if name == "config_similarity":
                return types.GetPromptResult(
                    description=f"Analyzing configuration similarity for {arguments['table_name']}",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Analyze the configuration similarity in the {arguments['table_name']} table:
1. Identify which configurations are most common across tenants
2. Group similar configurations together
3. Calculate what percentage of tenants use each configuration pattern
4. Provide recommendations for configuration standardization"""
                            )
                        )
                    ]
                )
            elif name == "config_outliers":
                return types.GetPromptResult(
                    description=f"Finding configuration outliers in {arguments['table_name']}",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Find configuration outliers in the {arguments['table_name']} table:
1. Identify configurations that differ significantly from the norm
2. Determine what percentage of tenants have these outlier configurations
3. Analyze the impact of these outlier configurations
4. Suggest whether these outliers should be standardized or kept as special cases"""
                            )
                        )
                    ]
                )
            elif name == "config_summary":
                return types.GetPromptResult(
                    description=f"Generating configuration summary for {arguments['table_name']}",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Generate a summary of configuration patterns in the {arguments['table_name']} table:
1. List the most common configuration settings
2. Identify trends in configuration usage
3. Highlight areas where configuration complexity could be reduced
4. Provide recommendations for configuration optimization"""
                            )
                        )
                    ]
                )
            else:
                raise ValueError(f"Unknown prompt: {name}")

    async def _execute_query(self, query: str) -> list[types.TextContent]:
        """Execute a read-only SQL query"""
        # Ensure query is read-only
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        try:
            async with await self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()

                    # Convert to list of dictionaries
                    results = [
                        dict(zip(columns, row))
                        for row in rows
                    ]

                    return [types.TextContent(
                        type="text",
                        text=json.dumps(results, default=str, indent=2)
                    )]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    async def _analyze_configurations(self, table_name: str, tenant_column: str, config_columns: List[str]) -> list[types.TextContent]:
        """Analyze configuration similarity across tenants"""
        config_columns_str = ", ".join(config_columns)
        query = f"""
            SELECT {tenant_column}, {config_columns_str}
            FROM {table_name}
            ORDER BY {tenant_column}
        """

        try:
            async with await self.get_connection() as conn:
                df = pd.read_sql(query, conn)

                # Create configuration signature
                df['config_signature'] = df[config_columns].apply(
                    lambda row: '_'.join(str(val) for val in row), axis=1
                )

                # Count occurrences of each configuration
                config_counts = df['config_signature'].value_counts()

                # Group tenants with same configuration
                config_groups = df.groupby('config_signature')[tenant_column].apply(list).to_dict()

                analysis = {
                    "total_tenants": len(df),
                    "unique_configurations": len(config_counts),
                    "most_common_configs": config_counts.head(10).to_dict(),
                    "config_groups": {
                        sig: {"count": len(tenants), "tenants": tenants[:10]}  # Show first 10 tenants
                        for sig, tenants in config_groups.items()
                    }
                }

                return [types.TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )]
        except Exception as e:
            logger.error(f"Error analyzing configurations: {e}")
            raise

    async def _find_deviations(self, table_name: str, config_column: str, threshold: float) -> list[types.TextContent]:
        """Find configurations that deviate from common patterns"""
        query = f"""
            SELECT {config_column}, COUNT(*) as count
            FROM {table_name}
            GROUP BY {config_column}
            ORDER BY count DESC
        """

        try:
            async with await self.get_connection() as conn:
                df = pd.read_sql(query, conn)
                total_records = df['count'].sum()

                # Calculate percentage for each value
                df['percentage'] = (df['count'] / total_records) * 100

                # Find deviations (values used by less than threshold%)
                deviations = df[df['percentage'] < threshold]

                analysis = {
                    "total_records": total_records,
                    "unique_values": len(df),
                    "deviations_found": len(deviations),
                    "deviations": deviations.to_dict(orient='records'),
                    "standard_values": df[df['percentage'] >= threshold].to_dict(orient='records')
                }

                return [types.TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )]
        except Exception as e:
            logger.error(f"Error finding deviations: {e}")
            raise

    async def _create_chart(self, query: str, chart_type: str, x_column: str, y_column: str, title: str, group_column: Optional[str] = None) -> list[types.ImageContent]:
        """Create a chart from query results"""
        try:
            async with await self.get_connection() as conn:
                df = pd.read_sql(query, conn)

                if chart_type == "bar":
                    if group_column:
                        fig = px.bar(df, x=x_column, y=y_column, color=group_column, title=title or "Bar Chart")
                    else:
                        fig = px.bar(df, x=x_column, y=y_column, title=title or "Bar Chart")
                elif chart_type == "pie":
                    fig = px.pie(df, values=y_column, names=x_column, title=title or "Pie Chart")
                elif chart_type == "line":
                    if group_column:
                        fig = px.line(df, x=x_column, y=y_column, color=group_column, title=title or "Line Chart")
                    else:
                        fig = px.line(df, x=x_column, y=y_column, title=title or "Line Chart")
                elif chart_type == "scatter":
                    if group_column:
                        fig = px.scatter(df, x=x_column, y=y_column, color=group_column, title=title or "Scatter Plot")
                    else:
                        fig = px.scatter(df, x=x_column, y=y_column, title=title or "Scatter Plot")
                elif chart_type == "heatmap":
                    # For heatmap, we need a pivot table
                    pivot_df = pd.pivot_table(df, values=y_column, index=x_column,
                                              columns=group_column if group_column else df.columns[2])
                    fig = go.Figure(data=go.Heatmap(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index))
                    fig.update_layout(title=title or "Heatmap")
                else:
                    raise ValueError(f"Unsupported chart type: {chart_type}")

                # Export chart to PNG
                img_buffer = io.BytesIO()
                fig.write_image(img_buffer, format='png')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                return [types.ImageContent(
                    type="image",
                    data=img_base64,
                    mimeType="image/png"
                )]
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            raise

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections for MCP protocol"""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    method = request.get("method")
                    params = request.get("params", {})
                    id = request.get("id")

                    # Map JSON-RPC methods to MCP handlers
                    if method == "initialize":
                        result = {
                            "capabilities": self.server.get_capabilities(
                                notification_options=NotificationOptions(),
                                experimental_capabilities={},
                            ),
                            "serverInfo": {
                                "name": "oracle-config-analysis",
                                "version": "0.1.0"
                            },
                            "protocolVersion": "2024-11-05"
                        }
                    elif method == "resources/list":
                        resources = await self.server.list_resources()
                        result = {"resources": resources}
                    elif method == "resources/read":
                        content = await self.server.read_resource(params["uri"])
                        result = {"contents": [{"type": "text", "text": content}]}
                    elif method == "tools/list":
                        tools = await self.server.list_tools()
                        result = {"tools": tools}
                    elif method == "tools/call":
                        content = await self.server.call_tool(params["name"], params["arguments"])
                        result = {"content": content}
                    elif method == "prompts/list":
                        prompts = await self.server.list_prompts()
                        result = {"prompts": prompts}
                    elif method == "prompts/get":
                        prompt_result = await self.server.get_prompt(params["name"], params["arguments"])
                        result = prompt_result
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    response = {
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": result
                    }

                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        }
                    }

                await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for {websocket.remote_address}")

    async def run_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Run WebSocket server for MCP protocol"""
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")
        await self.initialize_pool()

        async with serve(self.handle_websocket, host, port):
            await asyncio.Future()  # run forever

    async def run(self):
        """Run the MCP server with WebSocket support"""
        # Start WebSocket server in the background
        websocket_task = asyncio.create_task(self.run_websocket_server())

        # Also run stdio server for backward compatibility
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Starting Oracle Configuration Analysis MCP Server (stdio)")
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="oracle-config-analysis",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

async def main():
    """Main entry point"""
    # Get Oracle configuration from environment variables
    oracle_config = OracleConfig(
        host=os.getenv("ORACLE_HOST", "localhost"),
        port=int(os.getenv("ORACLE_PORT", "1521")),
        service_name=os.getenv("ORACLE_SERVICE_NAME", "ORCL"),
        user=os.getenv("ORACLE_USER", "system"),
        password=os.getenv("ORACLE_PASSWORD", "")
    )

    server = ConfigAnalysisServer(oracle_config)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())