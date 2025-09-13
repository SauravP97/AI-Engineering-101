from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather_status(location: str) -> str:
    """Get Weather details for the given location"""
    return "It's very sunny in San Francisco"


if __name__ == "__main__":
    mcp.run(transport="stdio")