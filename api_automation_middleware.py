
from transformers import pipeline
import pandas as pd
import sys
from helper.gradio_func import get_chart_data_closure
import gradio as gr
import torch
from langchain_ollama import ChatOllama
from helper.funcs import get_messages_api_automation_info
from custom_tool.tools import State, get_swagger_data
import os
from typing import Union, Dict

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()


tools = [get_swagger_data]
# LLM model : LLAMA, Claude
LLM_MODEL = os.getenv("LLM_MODEL")

llm = ChatOllama(model="llama3.2", temperature=0) if LLM_MODEL == "llama" else ChatAnthropic(
    model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# chatbot node


def chatbot(state: State):
    messages = get_messages_api_automation_info(state["messages"])
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
    print(f"result => {result}")
    return result.content


device = "cuda:0" if torch.cuda.is_available() else "mps"
print(f"\ncurrent device is {device}\n")
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                device=device)

# handle audio input, TODO: session or user_id setting


def run_agent(audio_input, chat_input, session_id: Union[str, None] = None, thread_id: Union[str, None] = "thread_1"):
    if audio_input is not None:
        # Process audio input
        transcription = pipe(audio_input, generate_kwargs={
                             "task": "transcribe"}, return_timestamps=True)["text"]
        config = {"configurable": {"thread_id": thread_id}}
        return transcription, stream_graph_updates(transcription, config)
    elif chat_input is not None and chat_input.strip() != "":
        # Process text input directly
        config = {"configurable": {"thread_id": thread_id}}
        # Return empty string for transcription if using text
        return "", stream_graph_updates(chat_input, config)
    else:
        raise gr.Error("No audio file or text input provided")


# WEB UI draw
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# state to closure


def get_data():
    return get_chart_data_closure(graph)


with gr.Blocks() as demo:
    gr.Markdown("""# Ciel AI Agent for api automation Agent with voice""")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], type="filepath")
            # New addition for keyboard chat
            chat_input = gr.Textbox(label="Type your message")
            transcript_output = gr.Textbox(label="Transcription")
            ai_response_output = gr.Textbox(label="AI Response")
#            ai_state_output = gr.Textbox(label="AI State")
            inputs = [audio_input, chat_input]  # Updated to include chat_input
            outputs = [
                transcript_output,
                ai_response_output,
            ]
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit")
        with gr.Column():
            data = gr.Textbox(label="data")

            @gr.render(inputs=data)
            def show_chart(text: str):
                if len(text) == 0:
                    gr.Markdown("")
                else:
                    CHARTDATA = StringIO(text)
                    df = pd.read_csv(CHARTDATA, sep=",")
                    gr.LinePlot(
                        df,
                        x='timestamp',
                        y='speed'
                    )

        # tiColumn="Ciel AI Agent: Transcribe Audio and Get AI Routing"
        # descriColumnon="Ciel the leading MOD, DRT Service Provider."
        # allow_Columngging="never"
#    ai_response_output.change(get_data_list, [], data)
    ai_response_output.change(get_data, [], data)
    # Clear both inputs
    clear_btn.click(lambda: None, None, [audio_input, chat_input])
    submit_btn.click(fn=run_agent, inputs=[
                     audio_input, chat_input], outputs=outputs, api_name="run_agent")
demo.launch(server_port=8080)
