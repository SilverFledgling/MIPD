#!/bin/bash
# ── Wait for SQL Server to start ────────────────────
echo "Waiting for SQL Server to start..."
sleep 15

# ── Run schema.sql ──────────────────────────────────
/opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" -C -i /docker-entrypoint-initdb.d/schema.sql

echo "MIPD Database initialized!"
