from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
from custom_tool.tools import State
from langgraph.prebuilt import ToolNode, tools_condition
from helper.funcs import get_messages_local_model_info
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
graph_builder = StateGraph(State)
llm = ChatOllama(model="llama3.2:latest", temperature=0)
tavily = TavilySearchResults(max_results=2)
tools = [tavily]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    messages = get_messages_local_model_info(state["messages"])
    message = llm_with_tools.invoke(messages)
    return {"messages": message}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        if "chatbot" in event:
            result = event["chatbot"]["messages"]
            print("Assistant:", result.content)
        else:
            print("=== message not for chatbot ====")


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input + "Exception", e)
        stream_graph_updates(user_input)
        break
    finally:
        # cleanup if needed
        pass
