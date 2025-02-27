import gradio as gr
import random
import time
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import Annotated
from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict, Literal
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    print(f"interrupt resumed ============================{human_response['data']}")
    return human_response["data"]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
tavily = TavilySearchResults(max_results=2)
tools = [tavily, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(State)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str, config: str):
    for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values"
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()
            result = event["messages"][-1]
            for item in result.content:
                if "name" in item and item["name"] == "human_assistance":
                    print(f"item name => {item['name']}")
                    human_command = Command(resume={"data": "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents."})
                    inner_events = graph.stream(human_command, config, stream_mode="values")
                    for e in inner_events:
                        if "messages" in e:
                            e["messages"][-1].pretty_print()
#            print(f"result => {result}")

    return [{"role": "user", "content": result.content}]
config = {"configurable": {"thread_id": "1"}}
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input, config)
    except:
        # fallback if input() is not available
        user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
        print("User: " + user_input)

        stream_graph_updates(user_input, config)
        break
