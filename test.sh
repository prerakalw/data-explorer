#!/bin/bash

# Test script for MCP client
# Update these values with your Oracle credentials
ORACLE_HOST="localhost"
ORACLE_PORT="1521"
ORACLE_SERVICE_NAME="ORCL"
ORACLE_USER="system"
ORACLE_PASSWORD="your_password"
SERVER_PATH="/path/to/mcp_oracle_config_server_websocket.py"

# Start the client
python mcp_client.py \
  --server "$SERVER_PATH" \
  --host "$ORACLE_HOST" \
  --port "$ORACLE_PORT" \
  --service-name "$ORACLE_SERVICE_NAME" \
  --user "$ORACLE_USER" \
  --password "$ORACLE_PASSWORD"
