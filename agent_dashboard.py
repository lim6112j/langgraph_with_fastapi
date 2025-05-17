import threading
import signal
from transformers import pipeline
import pandas as pd
import sys
import plotly.graph_objects as go
from helper.gradio_func import get_chart_data_closure
import gradio as gr
import torch
from langchain_ollama import ChatOllama
from helper.funcs import get_messages_info, get_messages_dashboard_info
from messages.messages import template
from custom_tool.tools import State, get_dashboard_info, get_chart_data
import os
from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import telebot
from threading import Thread


from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import interrupt


from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from langchain_core.utils.utils import secret_from_env
load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1",
                         openai_api_key=openai_api_key, **kwargs)


tools = [get_dashboard_info, get_chart_data]
# LLM model : LLAMA, Claude
LLM_MODEL = os.getenv("LLM_MODEL")

# llm = ChatOllama(model="llama3.2", temperature=0) if LLM_MODEL == "llama" else ChatAnthropic(
#    model="claude-3-5-sonnet-20240620")

llm = ChatOpenRouter(model_name="anthropic/claude-3.5-sonnet:20240620",)
# llm = ChatOllama(model="llama3.2:latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# chatbot node


def chatbot(state: State):
    messages = get_messages_dashboard_info(state["messages"])
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

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
graph = graph_builder.compile()

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
    gr.Markdown("""# Ciel AI Agent for Dashboard Agent with voice""")
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
# Telegram bot integration - moved to after Gradio launch to avoid hot-reload conflicts

# Telegram bot integration - moved to a separate function to avoid immediate execution


def start_telegram_integration():
    import os.path
    import time
    import threading

    # Get token from environment
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not found. Telegram bot not started.")
        return

    # Clean up any existing lock file at startup
    LOCK_FILE = "telegram_bot.lock"
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
            print("Cleaned up existing lock file at startup")
        except:
            pass

    # Check for existing lock file
    if os.path.exists(LOCK_FILE):
        try:
            # Force remove the lock file - we're starting fresh
            os.remove(LOCK_FILE)
            print("Removed existing lock file")
        except Exception as e:
            print(f"Error removing lock file: {e}")
            return

    # Create lock file
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(time.time()))
        print("Created Telegram bot lock file")
    except Exception as e:
        print(f"Error creating lock file: {e}")
        return

    # Initialize bot
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

    # Handler for text messages
    @bot.message_handler(content_types=['text'])
    def handle_text(message):
        try:
            user_input = message.text
            config = {"configurable": {
                "thread_id": f"telegram_{message.chat.id}"}}
            response = stream_graph_updates(user_input, config)
            bot.reply_to(message, response)
        except Exception as e:
            print(f"Error handling text message: {e}")
            bot.reply_to(message, f"Sorry, an error occurred: {str(e)}")

    # Handler for voice messages
    @bot.message_handler(content_types=['voice'])
    def handle_voice(message):
        try:
            # Download the voice message
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            # Save the voice message temporarily
            voice_file_path = f"temp_voice_{message.chat.id}.ogg"
            with open(voice_file_path, 'wb') as voice_file:
                voice_file.write(downloaded_file)

            # Process with whisper
            transcription = pipe(voice_file_path, generate_kwargs={
                "task": "transcribe"}, return_timestamps=True)["text"]

            # Get response from agent
            config = {"configurable": {
                "thread_id": f"telegram_{message.chat.id}"}}
            response = stream_graph_updates(transcription, config)

            # Send transcription and response
            bot.reply_to(
                message, f"Transcription: {transcription}\n\nResponse: {response}")

            # Clean up
            os.remove(voice_file_path)
        except Exception as e:
            print(f"Error processing voice message: {e}")
            bot.reply_to(message, f"Error processing voice message: {str(e)}")

    # Function to run the bot
    def run_bot():
        print("Starting Telegram bot...")
        try:
            # Use skip_pending=True to ignore messages that arrived while the bot was offline
            bot.infinity_polling(skip_pending=True, timeout=60)
        except Exception as e:
            print(f"Telegram bot error: {e}")
        finally:
            # Always clean up the lock file when the bot stops
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
                print("Removed lock file after bot stopped")

    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Store the bot and thread in global variables for cleanup
    global telegram_bot, telegram_bot_thread
    telegram_bot = bot
    telegram_bot_thread = bot_thread

    # We can't set signal handlers in a non-main thread, so we'll use a different approach
    print("Telegram bot started successfully")


# Initialize global variables
telegram_bot = None
telegram_bot_thread = None
telegram_lock_file = "telegram_bot.lock"

# Setup signal handlers for clean shutdown in the main thread


def signal_handler(sig, frame):
    print(f"Received signal {sig}, shutting down Telegram bot...")
    if 'telegram_bot' in globals() and telegram_bot:
        telegram_bot.stop_polling()
    if os.path.exists(telegram_lock_file):
        os.remove(telegram_lock_file)
    sys.exit(0)


# Register signal handlers in the main thread
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start the Telegram integration after a short delay
telegram_timer = threading.Timer(2.0, start_telegram_integration)
telegram_timer.daemon = True
telegram_timer.start()

# Launch Gradio app last, after setting up everything else
demo.launch(server_port=8080)
