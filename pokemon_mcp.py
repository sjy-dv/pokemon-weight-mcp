import httpx  
from mcp.server.fastmcp import FastMCP 

mcp = FastMCP("pokemon-weight-search")

async def fetch_pokemon_api_server(url: str):
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

@mcp.tool()
async def get_pokemon_weight(pokemon: str) -> str:
    """Return Pokemon Weigth for (pokemon name)"""
    url =f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}"
    data = await fetch_pokemon_api_server(url)
    return f"pokemon {pokemon} weight is {data['weight']/10}"


if __name__ == "__main__":
    mcp.run(transport="stdio")