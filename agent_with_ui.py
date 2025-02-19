from typing import Annotated, Union, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human"""
    human_response = interrupt(
        {
            "question": "Is this correct:",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, upddate the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # otherwise, receive information from the human receiver.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"
    # This time we explicitly update the state with a ToolMessage inside the tool
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
def stream_graph_updates(message: str, config: Dict):
    result: str
    for event in graph.stream(
            {"messages": [{"role": "user", "content": message}]},
            config,
            stream_mode="values",
    ):
        if "messages" in event:
                event["messages"][-1].pretty_print()
                result = event["messages"][-1]
    return result.content

from transformers import pipeline
import torch
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                device="mps")

def run_agent(audio_input, session_id: Union[str, None] = None, thread_id: Union[str, None] = "thread_1"):
    if audio_input is None:
        raise gr.Error("No audio file")
    transcription = pipe(audio_input, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]

    config = {"configurable": {"thread_id": thread_id}}

    return transcription, stream_graph_updates(transcription, config)

import gradio as gr

with gr.Blocks() as demo:
    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
    inputs=[audio_input]
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="AI Response"),
    ]
    clear_btn = gr.Button("Clear")
    submit_btn = gr.Button("Submit")
    clear_btn.click(lambda :None, None, audio_input)
    submit_btn.click(fn=run_agent, inputs= inputs, outputs=outputs, api_name="run_agent")
   # title="Ciel AI Agent: Transcribe Audio and Get AI Routing"
   # description="Ciel the leading MOD, DRT Service Provider."
   # allow_flagging="never"

    with gr.Row():
        btn1 = gr.Button("hello")

demo.launch()
