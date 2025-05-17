from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
from custom_tool.tools import State, get_data_from_site
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(State)
llm = ChatOllama(model="llama3.2:latest", temperature=0)

tools = [get_data_from_site]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}


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
            print("No messages in event:", event)


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
