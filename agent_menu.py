from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import interrupt


from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

import os
from custom_tool.tools import State, get_menus
from messages.messages import template
from helper.funcs import get_messages_info, get_messages_menu_info

# outes data schema example
routes = [{"legs": [{"steps": [{"maneuver": {"location": [1, 2]}}] }]}]

tools = [get_menus]
# LLM model : LLAMA, Claude
from langchain_ollama import ChatOllama
LLM_MODEL = os.getenv("LLM_MODEL")

llm = ChatOllama(model="llama3.2", temperature=0) if LLM_MODEL == "llama" else ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# chatbot node
def chatbot(state: State):
    messages = get_messages_menu_info(state["messages"])
#    print(f"\nstate[messages] => {state['messages']}\n")
#    print(f"\n[messages] => {messages}\n")
    message = llm_with_tools.invoke(messages)
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# link nodes
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

# helper function for run_agent
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
device = "cuda:0" if torch.cuda.is_available() else "mps"
print(f"\ncurrent device is {device}\n")
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                device=device)

# handle audio input, TODO: session or user_id setting
def run_agent(audio_input, session_id: Union[str, None] = None, thread_id: Union[str, None] = "thread_1"):
    if audio_input is None:
        raise gr.Error("No audio file")
    transcription = pipe(audio_input, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    config = {"configurable": {"thread_id": thread_id}}
    return transcription, stream_graph_updates(transcription, config)

# WEB UI draw
import gradio as gr
from helper.gradio_func import filter_map
import plotly.graph_objects as go
from helper.gradio_func import draw_route_list_closure, get_menu_list_closure, get_image_from_url
import json

# draw route from list
def get_menu_list():
    return get_menu_list_closure(graph)

with gr.Blocks() as demo:
    gr.Markdown("""# Ciel AI Agent for Menu-pan with voice""")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
            transcript_output = gr.Textbox(label="Transcription")
            ai_response_output = gr.Textbox(label="AI Response")
            inputs=[audio_input]
            outputs=[
                transcript_output,
                ai_response_output,
            ]
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit")
        # tiColumn="Ciel AI Agent: Transcribe Audio and Get AI Routing"
        # descriColumnon="Ciel the leading MOD, DRT Service Provider."
        # allow_Columngging="never"
        with gr.Column():
            menus = gr.Textbox(label="Menus")
            @gr.render(inputs=menus)
            def show_images(text: str):
                if len(text) == 0:
                    gr.Markdown("")
                else:
                    json_obj = json.loads(text)
#                    print(f"gradio read menus: {json_obj[0]}")
                    for obj in json_obj:
                        img_url = obj['url']
                        img = get_image_from_url(img_url)
                        gr.Image(value = img)
    ai_response_output.change(get_menu_list, [], menus)
    clear_btn.click(lambda :None, None, audio_input)
    submit_btn.click(fn=run_agent, inputs= inputs, outputs=outputs, api_name="run_agent")
demo.launch(server_port=8080)
