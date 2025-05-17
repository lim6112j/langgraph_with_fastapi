import os
import sys
import json
import uuid
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory  # Updated import
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, List, Optional
from typing_extensions import TypedDict
from custom_tool.tools import *  # Import your tools here

# Define State class to match the one in tools.py


class State(TypedDict):
    messages: Annotated[list, add_messages]
    menus: list
    routes: list
    chart_data: list


def main():
    # Initialize the language model with error handling
    try:
        # Try llama3 first (most likely to be available)
        llm = ChatOllama(model="llama3.2:latest")
    except Exception as e:
        print(f"Error loading llama3: {e}")
        try:
            # Fall back to llama2 if llama3 is not available
            llm = ChatOllama(model="llama2")
        except Exception as e:
            print(f"Error loading llama2: {e}")
            print(
                "Please ensure Ollama is installed and running with either llama3 or llama2 models.")
            sys.exit(1)

    # Initialize state for tools
    current_state = State(
        messages=[],
        menus=[],
        routes=[],
        chart_data=[]
    )

    # Create wrapper functions for our custom tools that handle state
    def routes_wrapper(start: str, destination: str) -> str:
        """Get route information from start to destination.

        Args:
            start: The starting location name
            destination: The destination location name

        Returns:
            Information about available routes
        """
        tool_call_id = str(uuid.uuid4())
        try:
            result = get_routes(current_state, start,
                                destination, tool_call_id)
            # If it's a Command, handle it to update the state
            if hasattr(result, 'update'):
                for key, value in result.update.items():
                    if key in current_state:
                        current_state[key] = value
                # Return a string representation of the routes
                if 'routes' in result.update and result.update['routes']:
                    routes_data = result.update['routes']
                    return f"Routes from {start} to {destination}:\n{json.dumps(routes_data, indent=2)}"
            return str(result)
        except Exception as e:
            return f"Error getting routes: {str(e)}"

    def menus_wrapper() -> str:
        """Get restaurant menu information.

        Returns:
            Available menu items
        """
        tool_call_id = str(uuid.uuid4())
        try:
            result = get_menus(current_state, tool_call_id)
            # If it's a Command, handle it to update the state
            if hasattr(result, 'update'):
                for key, value in result.update.items():
                    if key in current_state:
                        current_state[key] = value
                # Return a string representation of the menus
                if 'menus' in result.update and result.update['menus']:
                    return f"Available menus:\n{result.update['menus']}"
            return str(result)
        except Exception as e:
            return f"Error getting menus: {str(e)}"

    def web_data_wrapper(keyword: str) -> str:
        """Get data from a website based on keyword.

        Args:
            keyword: The search term to use

        Returns:
            Information retrieved from the website
        """
        try:
            result = get_data_from_site(current_state, keyword)
            # For this tool, we don't need to update state as it returns directly
            return f"Data for '{keyword}':\n{result}"
        except Exception as e:
            return f"Error getting web data: {str(e)}"

    # Create LangChain tools using our wrapper functions
    tools = [
        Tool.from_function(
            func=routes_wrapper,
            name="get_routes",
            description="Get route information between two locations. Requires start and destination parameters."
        ),
        Tool.from_function(
            func=menus_wrapper,
            name="get_menus",
            description="Get menu information from restaurants. No parameters needed."
        ),
        Tool.from_function(
            func=web_data_wrapper,
            name="get_data_from_site",
            description="Search for information on a website using a keyword. Requires keyword parameter."
        )
    ]

    # Create a memory for the agent
    memory = ConversationBufferMemory(return_messages=True)

    # Format tools for prompt template
    tool_names = ", ".join([tool.name for tool in tools])
    tools_formatted = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools])

    # Create a prompt template for ReAct agents with tools variable
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant with access to tools that can help with navigation, 
finding restaurants, and other useful tasks. 

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in JSON format
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Available tools:
{tools_formatted}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create an agent using ReAct framework which works better with Ollama models
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create an agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )

    try:
        # Run the agent (you can customize the input as needed)
        # We no longer need to pass state as the wrappers handle it
        response = agent_executor.invoke({
            "input": "Show me routes from Seoul to Busan"
        })
        print(response["output"])
    except Exception as e:
        print(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
