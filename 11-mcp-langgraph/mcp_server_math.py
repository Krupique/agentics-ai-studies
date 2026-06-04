from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.prompt()
def example_prompt(question: str) -> str:
    # Returns the prompt template for the math wizard
    return f"""
    You are a math assistant. Answer the question.
    Question: {question}
    """

@mcp.prompt()
def system_prompt() -> str:
    # Returns the prompt template to the system
    return """
    You are an AI assistant, use the tools if necessary.
    """

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    # Returns a personalized greeting
    return f"Hello, {name}!"

@mcp.resource("config://app")
def get_config() -> str:
    # Returns the application configuration (example)
    return "App Settings"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add 2 numbers"""
    # Calculates and returns the sum of two integers.
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    # Calcula e retorna o produto de dois inteiros
    return a * b

if __name__ == "__main__":
    # Starts the MCP execution loop via STDIO.
    mcp.run()
