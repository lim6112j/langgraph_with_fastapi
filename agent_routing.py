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


from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

df = pd.read_csv("./data/locations.csv")
names = df["name"].tolist()
lons = df["longitude"].tolist()
lats = df["latitude"].tolist()
loc_dict = {names[i]: (lons[i], lats[i]) for i in range(0, len(names))}
@tool
def get_routes(start: str, destination: str) -> str:
    """ Get route information from start to destination."""
    start_loc = str(loc_dict[start][0]) + ',' + str(loc_dict[start][1])
    destination_loc = str(loc_dict[destination][0]) + ',' + str(loc_dict[destination][1])
    locations = start_loc + ';' + destination_loc
    print(locations)
    import os
    import http.client
    OSRM_API = os.getenv("OSRM_API")
    # http://localhost:5001/route/v1/driving/127.919323,36.809656;128.080629,36.699223?steps=true
    conn = http.client.HTTPConnection(OSRM_API, 5001)
    conn.request("GET", "/route/v1/driving/" + locations + "?steps=true")
    res = conn.getresponse()
    routes = res.read()
    conn.close()
    print(f"osrm response routes data {routes}")
    if not routes:
        return "No routes found."
    # response = "\n".join([f"{route['id']}: {route['title']}" for route in routes])
    return f"Available routes:\n{routes}"

tool = TavilySearchResults(max_results=2)
tools = [get_routes]
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

# airbnb data
from datasets import load_dataset
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv("./data/locations.csv")

def filter_map():
    names = df["name"].tolist()
    lons = df["longitude"].tolist()
    lats = df["latitude"].tolist()
    loc_list = [(names[i], (lons[i], lats[i])) for i in range(0, len(names))]
    # map figure
    fig = go.Figure(go.Scattermapbox(
        customdata=loc_list,
        lat=lats,
        lon=lons,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),
        hoverinfo="text",
        hovertemplate='<b>Name</b>: %{customdata[0]}<br><b>Loc</b>: $%{customdata[1]}'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=37.497467,
                lon=127.027458
            ),
            pitch=0,
            zoom=9
        ),
    )
    return fig
# ui draw
with gr.Blocks() as demo:
    gr.Markdown("""# Ciel AI Agent for routing with voice""")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
            inputs=[audio_input]
            outputs=[
                gr.Textbox(label="Transcription"),
                gr.Textbox(label="AI Response"),
            ]
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit")

        # tiColumn="Ciel AI Agent: Transcribe Audio and Get AI Routing"
        # descriColumnon="Ciel the leading MOD, DRT Service Provider."
        # allow_Columngging="never"
        with gr.Column():
            min_price = gr.Number(value=250, label="Minimum Price")
            max_price = gr.Number(value=1000, label="Maximum Price")
            boroughs = gr.CheckboxGroup(choices=["Queens", "Brooklyn", "Manhattan", "Bronx", "Staten Island"], value=["Queens", "Brooklyn"], label="Select Boroughs:")
            btn = gr.Button(value="Update Filter")
            map = gr.Plot()
    clear_btn.click(lambda :None, None, audio_input)
    submit_btn.click(fn=run_agent, inputs= inputs, outputs=outputs, api_name="run_agent")
    demo.load(filter_map, [], map)
    btn.click(filter_map, [], map)

demo.launch()
