# Oracle Configuration Analysis MCP Server

This MCP server connects to your Oracle database and provides tools for analyzing configuration data across your multi-tenant application.

## Features

1. **Database Connection**: Connects to Oracle using connection pooling for efficient resource management
2. **Resource Exposure**: Exposes configuration tables as resources
3. **Read-Only Queries**: Allows executing SELECT queries on the database
4. **Configuration Analysis**: Tools for analyzing configuration similarity and deviations
5. **Data Visualization**: Creates charts from query results
6. **Analysis Prompts**: Pre-built prompts for common analysis tasks

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export ORACLE_HOST=your-oracle-host
export ORACLE_PORT=1521
export ORACLE_SERVICE_NAME=ORCL
export ORACLE_USER=your-username
export ORACLE_PASSWORD=your-password
```

3. Configure Claude Desktop:
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "oracle-config-analysis": {
      "command": "python",
      "args": ["/path/to/mcp_oracle_config_server.py"],
      "env": {
        "ORACLE_HOST": "your-oracle-host",
        "ORACLE_PORT": "1521",
        "ORACLE_SERVICE_NAME": "ORCL",
        "ORACLE_USER": "your-username",
        "ORACLE_PASSWORD": "your-password"
      }
    }
  }
}
```

## Available Tools

### 1. execute_query
Executes read-only SQL queries (SELECT only).

### 2. analyze_configurations
Analyzes configuration similarity across tenants by grouping similar configurations.

### 3. find_deviations
Identifies configurations that deviate from common patterns based on usage percentage.

### 4. create_chart
Creates visualizations (bar, pie, line, scatter, heatmap) from query results.

## Available Prompts

1. **config_similarity**: Analyzes how similar configurations are across different tenants
2. **config_outliers**: Finds configurations that differ significantly from the norm
3. **config_summary**: Generates a comprehensive summary of configuration patterns

## Security Notes

- Only SELECT queries are allowed to ensure data safety
- Connection pooling is used for efficient resource management
- Oracle Instant Client is required for operation

## Usage Examples

```
# Analyze configuration similarity
analyze_configurations(
    table_name="APP_CONFIG",
    tenant_column="COMPANY_ID",
    config_columns=["SETTING_A", "SETTING_B", "SETTING_C"]
)

# Find configuration deviations
find_deviations(
    table_name="APP_CONFIG",
    config_column="SETTING_VALUE",
    threshold=20
)

# Create a configuration usage chart
create_chart(
    query="SELECT SETTING_NAME, COUNT(*) as USAGE_COUNT FROM APP_CONFIG GROUP BY SETTING_NAME",
    chart_type="bar",
    x_column="SETTING_NAME",
    y_column="USAGE_COUNT",
    title="Configuration Usage Distribution"
)
```
