import os
from langchain.agents import ToolNode, ChatbotNode
from langchain.llms import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import LangchainToolkit
from custom_tool.tools import *  # Import your tools here

def main():
    # Initialize the language model
    llm = ChatOllama(model="llama3.2")

    # Define your tools
    tools = [get_routes, get_menus, get_data_from_site]  # Add your specific tools here

    # Create a toolkit
    toolkit = LangchainToolkit(embeddings=OpenAIEmbeddings(), tools=tools)

    # Create a memory for the chatbot
    memory = ConversationBufferMemory()

    # Create the chatbot node
    chatbot_node = ChatbotNode(llm=llm, memory=memory)

    # Create the tool node
    tool_node = ToolNode(tools=tools)

    # Combine nodes into an agent executor
    agent = AgentExecutor.from_nodes([chatbot_node, tool_node])

    # Run the agent (you can customize the input as needed)
    response = agent.run("Your input message here")
    print(response)

if __name__ == "__main__":
    main()
