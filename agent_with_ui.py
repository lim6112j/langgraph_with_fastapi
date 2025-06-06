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

# airbnb data
from datasets import load_dataset
import plotly.graph_objects as go

dataset = load_dataset("gradio/NYC-Airbnb-Open-Data", split="train")
df = dataset.to_pandas()

def filter_map(min_price, max_price, boroughs):
    new_df = df[(df['neighbourhood_group'].isin(boroughs)) &
            (df['price'] > min_price) & (df['price'] < max_price)]
    names = new_df["name"].tolist()
    prices = new_df["price"].tolist()
    text_list = [(names[i], prices[i]) for i in range(0, len(names))]
    # map figure
    fig = go.Figure(go.Scattermapbox(
        customdata=text_list,
        lat=new_df['latitude'].tolist(),
        lon=new_df['longitude'].tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),
        hoverinfo="text",
        hovertemplate='<b>Name</b>: %{customdata[0]}<br><b>Price</b>: $%{customdata[1]}'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=40.67,
                lon=-73.90
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
    demo.load(filter_map, [min_price, max_price, boroughs], map)
    btn.click(filter_map, [min_price, max_price, boroughs], map)

demo.launch()
