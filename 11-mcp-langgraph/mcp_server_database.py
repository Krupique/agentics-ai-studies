import sqlite3
import json
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional 

mcp = FastMCP("Clients")

conn = sqlite3.connect("crm_clients.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS tb_clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
)
""")
conn.commit()

# --- Pydantic Models for Validation ---
class BaseClient(BaseModel):
    """Basic template for a client, used for creation input."""
    name: str = Field(..., min_length=1, description="Client name")
    email: EmailStr = Field(..., description="Client email")

class DBClient(BaseClient):
    """A template for a client as stored in the database includes the ID."""
    id: int = Field(..., description="Unique customer ID")

class ResponseClient(BaseModel):
    """Template for customer responses, may include data or errors.."""
    id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    error: Optional[str] = None

class ListResponseClient(BaseModel):
    """Template for a list of all clients.."""
    clients: List[DBClient]


# --- MCP Tools with Pydantic Validation ---

@mcp.tool()
def create_customer(cliente_data: BaseClient) -> str:
    # Documentation: Register a client with their name and email address.
    """
        Register a customer with name and email.
        Expects a JSON object with 'name' (string) and 'email' (valid email string).
        Returns the created customer data in JSON format or an error.
    """
    try:
        # The data is already validated by Pydantic through the type annotation
        # Inserts the new client into the tb_clients table
        cursor.execute(
            "INSERT INTO tb_clients (name, email) VALUES (?, ?)",
            (cliente_data.name, cliente_data.email)
        )
        conn.commit()
        customer_id = cursor.lastrowid
        response_data = DBClient(id=customer_id, name=cliente_data.name, email=cliente_data.email)
        return response_data.model_dump_json()
    except sqlite3.IntegrityError: # Capture duplicate email error, for example.
        error_response = ResponseClient(error="Error creating client: email already exists or invalid data.")
        return error_response.model_dump_json()
    except Exception as e:
        error_response = ResponseClient(error=f"Unexpected error: {str(e)}")
        return error_response.model_dump_json()

@mcp.tool()
def get_customer(customer_id: int) -> str:
    # Documentation: Read customer data by ID
    """
        Read customer data by ID.
        Expects an 'customer_id' (integer).
        Returns customer data in JSON format or an error if not found.
    """
    if not isinstance(customer_id, int) or customer_id <= 0:
        error_response = ResponseClient(error="Invalid customer ID. Must be a positive integer.")
        return error_response.model_dump_json()

    cursor.execute(
        "SELECT id, name, email FROM tb_clients WHERE id = ?", (customer_id,)
    )
    row = cursor.fetchone()

    if row:
        cliente = DBClient(id=row[0], name=row[1], email=row[2])
        return cliente.model_dump_json()
    
    error_response = ResponseClient(error="Customer not found")
    return error_response.model_dump_json()

@mcp.tool()
def list_clients() -> str:
    # Documentation: List all clients
    """
        List all customers.
        Returns a list of all customers in JSON format.
    """
    cursor.execute("SELECT id, name, email FROM tb_clients")
    rows = cursor.fetchall()
    db_clients = [DBClient(id=r[0], name=r[1], email=r[2]) for r in rows]
    response_data = ListResponseClient(clients=db_clients)
    return response_data.model_dump_json()

# Checks if the script is being executed directly
if __name__ == "__main__":

    # Starts the MCP server with HTTP streaming transport (must be used when running the app_langgraph.py script)
    mcp.run(transport="streamable-http")

    # Starts the MCP server with SSE transport (must be used when running the app_pydantic_ai.py script)
    # mcp.run(transport="sse")



    