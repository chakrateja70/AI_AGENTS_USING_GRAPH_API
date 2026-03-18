import os
import operator
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated, Optional, Literal

from langchain_tavily import TavilySearch
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

load_dotenv(override=True)

# Initialize the OpenAI client
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

#chat state
class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    result: Optional[str]
    user_input: Optional[str]
    llm_calls: int

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def tavily_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs fresh or real-time web information.
    It searches the web with Tavily and returns the top results for the query as a string.
    """
    tavily_run = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_run.invoke(query)

@tool
def arxiv_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs academic or research-paper sources.
    It searches arXiv for papers matching the query and returns top results as a string.
    """
    arxiv_wrapper = ArxivAPIWrapper( #type: ignore
        top_k_results=2,
        doc_content_chars_max=1024
    )
    arxiv_run = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    return arxiv_run.invoke(query)

# Augment the LLM with tools
tools = [multiply, tavily_tool_run, arxiv_tool_run]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# Define the LLM call node
def llm_call(state: ChatState) -> ChatState:
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# Define the tool call node
def tool_node(state: ChatState) -> ChatState:
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Define the graph
def should_continue(state: ChatState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(ChatState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show and save the agent graph image
graph_png = agent.get_graph(xray=True).draw_mermaid_png()
graph_path = os.path.join(os.getcwd(), "agent_graph.png")
with open(graph_path, "wb") as f:
    f.write(graph_png)
print(f"Graph image saved to: {graph_path}")

# Invoke
messages = [HumanMessage(content="what is latest news about iran and israel?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()