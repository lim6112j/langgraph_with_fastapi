from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.types import Command

class State(TypedDict):
    messages: Annotated[list, add_messages]
    routes: list

from langchain_core.messages import ToolMessage, SystemMessage
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
import json
import os
df = pd.read_csv("./data/locations.csv")
names = df["name"].tolist()
lons = df["longitude"].tolist()
lats = df["latitude"].tolist()
loc_dict = {names[i]: (lons[i], lats[i]) for i in range(0, len(names))}
routes = [{"legs": [{"steps": [{"maneuver": {"location": [1, 2]}}] }]}]
@tool
def get_routes(state: State, start: str, destination: str,  tool_call_id: Annotated[str, InjectedToolCallId]):
    """ Get route information from start to destination.
    after gather start and destination from user,
    call this function for route data
    Args:
       start (str): start location name
       destination (str): destination location name
    """
    start_loc = str(loc_dict[start][0]) + ',' + str(loc_dict[start][1])
    destination_loc = str(loc_dict[destination][0]) + ',' + str(loc_dict[destination][1])
    locations = start_loc + ';' + destination_loc
#    print(locations)

    import http.client
    OSRM_API = os.getenv("OSRM_API")
    # http://localhost:5001/route/v1/driving/127.919323,36.809656;128.080629,36.699223?steps=true
    conn = http.client.HTTPConnection(OSRM_API, 5001)
    conn.request("GET", "/route/v1/driving/" + locations + "?steps=true")
    res = conn.getresponse()
    routes_bytes = res.read()
    conn.close()
#    print(f"osrm response routes data {routes}")
    if not routes_bytes:
        return "No routes found."
    # response = "\n".join([f"{route['id']}: {route['title']}" for route in routes])

    json_str = routes_bytes.decode('utf8').replace("'", '"')
    json_routes = json.loads(json_str)
    routes = json_routes['routes']

#    print(json_routes['routes'])
#    print(f"\nlength of routes : {len(json_routes['routes'])}")
#    fig = draw_route(loc_dict[start], loc_dict[destination])
#    fig.show()
#    return f"Available routes:\n{routes}"
    return Command(
         update={
             "routes": routes,
             "messages": [
                 ToolMessage(routes_bytes, tool_call_id=tool_call_id)
             ],
         }
    )

tool = TavilySearchResults(max_results=2)
tools = [get_routes]
from langchain_ollama import ChatOllama
LLM_MODEL = os.getenv("LLM_MODEL")

llm = ChatOllama(model="llama3.2", temperature=0) if LLM_MODEL == "llama" else ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# template = """Your job is to get information from a user about two location and to explain the route with tool `get_routes`

# You should get the following information from them:

# - What the objective of the prompt is
# - What variables will be passed into the prompt template
# - Any constraints for what the output should NOT do
# - Any requirements that the output MUST adhere to

# If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

# After you are able to discern all the information, call the relevant tool."""

template = """You are a car navigation guide, Your job is to get information from a user about two location and to explain the route with tool `get_routes`

If you are not able to discern two location name, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

def chatbot(state: State):
    messages = get_messages_info(state["messages"])
#    print(f"\nstate[messages] => {state['messages']}\n")
#    print(f"\n[messages] => {messages}\n")
    message = llm_with_tools.invoke(messages)
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
def draw_route(start, destination):

    fig = go.Figure().add_trace(go.Scattermapbox(
        mode='lines',
        lon = [start[0], destination[0]],
        lat = [start[1], destination[1]],
        line_color='green',
        name='routes calculated'
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
            zoom=15
        ),
    )
    return fig
def draw_route_list():
# loc_list : routes
    config = {"configurable": {"thread_id": "thread_1"}}
    state = graph.get_state(config)



    try:
        routes = state.values['routes']
        print(f"\nstate => {routes}\n")
        loc_list = routes[0]
        print(f"routes[0] : {loc_list}")
        lons = []
        lats = []
        for leg in loc_list['legs']:
            for step in leg['steps']:
                #           print(f"maneuver data : {step[''maneuver']}")
                location = (step['maneuver'])['location']
                lons.append(location[0])
                lats.append(location[1])
                print(f"\nlats from routes data => {lats}\n")
                fig = go.Figure().add_trace(go.Scattermapbox(
                    mode='lines',
                    lon = lons,
                    lat = lats,
                    line_color='green',
                    name='routes calculated'
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
                        zoom=13
                    ),
                )
        return fig
    except:
        print("routes not ready")
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
            zoom=13
        ),
    )
    return fig
# ui draw
with gr.Blocks() as demo:
    gr.Markdown("""# Ciel AI Agent for routing with voice""")
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
            btn = gr.Button(value="Update Filter")
            map = gr.Plot()
    ai_response_output.change(draw_route_list, [], map)
    clear_btn.click(lambda :None, None, audio_input)
    submit_btn.click(fn=run_agent, inputs= inputs, outputs=outputs, api_name="run_agent")
    demo.load(filter_map, [], map)
    btn.click(draw_route_list, [], map)

demo.launch()
