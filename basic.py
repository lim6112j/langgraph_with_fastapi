import gradio as gr
import random
import time
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from typing import Annotated
from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)
llm = ChatOllama(model="llama3.2")
tools = []
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            result = value["messages"][-1]
    return "", [{"role": "user", "content": result.content}]
def run_agent(user_input: str, chatbot_history):
    print(f"user message: {user_input}")
    try:
        return stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        raise
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(run_agent, [msg, chatbot], [msg, chatbot])

demo.launch()
