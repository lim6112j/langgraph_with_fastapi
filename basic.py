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
load_dotenv()
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    you can get current time. use this tool when you need to
    """
    from datetime import datetime
    c_time = datetime.now().strftime(format)
    print(f"current time : {c_time}")
    return c_time
@tool
def get_user_location():
    """you can get current user's location"""
    return "Seoul, South Korea"

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    print(f"interrupt resumed ============================{human_response['data']}")
    return human_response["data"]

llm = ChatOllama(model="llama3.2")
#llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
tools = [get_now, get_user_location, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    message = llm_with_tools.invoke(state["messages"])
#    assert len(message.tool_calls) <= 1
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

def stream_graph_updates(user_input: str, config: Dict):
    for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values"
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()
            result = event["messages"][-1]
    return "", [{"role": "user", "content": result.content}]
def run_agent(user_input: str, chatbot_history, thread_id: Union[str, None]="thread_1"):
    print(f"user message: {user_input}")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        return stream_graph_updates(user_input, config)
    except:
        # fallback if input() is not available
        raise
def expert_talk_closure(talk, gra, conf):
    g = gra
    def expert_talk_inner():
        human_command = g.invoke(Command(resume={"data": talk}), config=conf)
        return talk
    print(f"after human_assistance state =>  {g.get_state(conf)}")
    return expert_talk_inner()

def expert_talk(talk: str):
    config = {"configurable": {"thread_id": "thread_1"}}
    return expert_talk_closure(talk,graph,config)
with gr.Blocks() as demo:

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    expert = gr.Textbox(label="Expert")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(run_agent, [msg, chatbot], [msg, chatbot])
    expert.submit(expert_talk, [expert], [msg])
demo.launch()
