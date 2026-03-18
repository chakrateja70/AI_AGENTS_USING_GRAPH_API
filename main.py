import os
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#chat state
class ChatState(TypedDict, total=False):
    messages: List[dict]
    result: Optional[str]
    user_input: Optional[str]

#LLM Model
def openai_llm(conversation_text: str) -> str:
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": conversation_text
            }
        ]
    )
    return response.output[0].content[0].text

#Node 1: User Input
def user_input_node(state: ChatState) -> ChatState:
    messages = state.get("messages", [])
    user_text = state.get("user_input")
    messages.append("user: " + user_text)
    return {**state, "messages": messages}

#Node 2: Agent Node(OpenAI)
def agent_node(state: ChatState) -> ChatState:
    messages = state.get("messages", [])
    conversation_text = "\n".join(messages)
    response = openai_llm(conversation_text)
    messages.append("Assistant: " + response)
    return {**state, "messages": messages, "result": response}

#Graph (User -> agent -> END)
graph = StateGraph(ChatState)
graph.add_node("user", user_input_node)
graph.add_node("agent", agent_node)

graph.set_entry_point("user")
graph.add_edge("user", "agent")
graph.add_edge("agent", END)
app = graph.compile()

#Run the graph
final_state = app.invoke({"user_input": "what is my previous question?"})
print("final state keys", list(final_state.keys()))
print(final_state["result"])