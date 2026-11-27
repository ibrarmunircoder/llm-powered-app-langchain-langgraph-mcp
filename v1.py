from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily = TavilyClient()

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet

    Args:
        query: The query to search for

    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatBedrockConverse(
    model="qwen.qwen3-coder-30b-a3b-v1:0",
    region_name="us-east-1",
)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    )
    print(result)
    print("Hello from llm-powered-apps-with-langchain-langgraph-mcp-course!")


if __name__ == "__main__":
    main()