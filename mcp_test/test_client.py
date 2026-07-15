import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_env = {
        **os.environ,
        "DATAHUB_GMS_URL": os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080"),
        "DATAHUB_GMS_TOKEN": os.environ["DATAHUB_GMS_TOKEN"],
        "TOOLS_IS_MUTATION_ENABLED": "true",
    }

    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-datahub@latest"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"Connected. {len(tools.tools)} tools available:\n")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            print("\nCalling 'search' for 'synaptoflow'...\n")
            result = await session.call_tool("search", {"query": "synaptoflow"})
            print(result)


if __name__ == "__main__":
    asyncio.run(main())