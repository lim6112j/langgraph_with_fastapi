from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
# ollamachat model
from langchain_ollama import ChatOllama


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = ChatOllama(model="llama3.2:latest")  # Updated initialization


def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.compile()
# draw the graph
graph_builder.draw_graph("chatbot.png")
