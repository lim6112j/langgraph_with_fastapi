import torch
from transformers import pipeline
import ollama
import gradio as gr
import os
import pandas as pd
import io
import tools
def create_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    print(f"File '{filename}' created with content: {content}")

def read_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        print(f"Content of '{filename}': {content}")
        return content
    except fileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def delete_file(filename):
    try:
        os.remove(filename)
        print(f"File '{filename}' deleted")
    except FileNotFoundError:
        print(f"File '{filename}' not found")

def get_response_with_tools(prompt):
    tools = tools.tools
    response = ollama.chat(
        model='llama3.2',
        message=[{'role': 'user', 'content': prompt}],
        tools=tools,
    )
    tool_calls = response['message']['tool_calls']
    results = []
    task_table = None
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        arguments = tool_call('function')['arguments']
