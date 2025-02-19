import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
device = "mps"
torch_dtype = torch.float16
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage = True, use_safetensors = True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model = model,
    tokenizer = processor.tokenizer,
    feature_extractor = processor.feature_extractor,
    torch_dtype = torch_dtype,
    device = device,
)
# result = pipe("./test-audio-file2.mp3")
# print(result["text"])

# interaction with llm
import ollama

def get_response(prompt):
    response = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

import pyaudio
import wave

def record_audio(filename="prompt.mp3", duration=3, sample_rate=44100, channels=1, chunk=1024):
    """
    Record audio from the microphone and save it to a file

    :param filename: Name of the output file (default: "prompt.mp3")
    :param duration: Duration of the recording in seconds (default: 5)
    :param sample_rate: Sample rate of the recording (default: 44100 Hz)
    :param channels: Number of audio channels (default: 2 for stereo)
    :param chunk: Number of frames per buffer (default: 1024)
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("Recording....")
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # save the recorded data as a wave file
    wf = wave.open(filename.replace('.mp3', '.wav'), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved as {filename.replace('.mp3', '.wav')}")



def transcribe(audio_filepath):
    result = pipe(audio_filepath)
    return result["text"]


# record_audio()
# prompt = transcribe("./prompt.wav")
# print(prompt)
# print(get_response(prompt))


# task
import pandas as pd
from datetime import datetime
import tools
# create an empty data frame for task database
tasks_df = pd.DataFrame(columns=['task', 'status', 'creation_date', 'completed_date'])
# print(tasks_df)

# Tool for adding a task
def add_task(task_description):
    """
    Add a task to the tasks database.
    """
    new_task = pd.DataFrame({
        'task': [task_description],
        'status': ['Not Started'],
        'creation_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'completed_date': [None]
    })
    global tasks_df
    tasks_df = pd.concat([tasks_df, new_task], ignore_index=True)

    return tasks_df

# create tasks in a backlog task db
tool_add_tasks_to_db = tools.tools[3]
# print(tool_add_tasks_to_db)

def get_response_with_tools(prompt):
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        tools=[tool_add_tasks_to_db])
    # process tool calls if present
    if 'tool_calls' in response['message']:
        for tool_call in response['message']['tool_calls']:
            if tool_call['function']['name'] == 'add_task':
                task_description = tool_call['function']['arguments']['task_description']
                add_task(task_description)
                print(f"Task added: {task_description}")
    else:
        return response['message']['content']

# get_response_with_tools("Create a task to create a local voice AI assistant")
# print(tasks_df)

import os
import http.client
import urllib.parse
def create_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    return f"File {filename} created successfully"

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def edit_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    return f"File {filename} edited successfully"

def delete_file(filename):
    os.remove(filename)
    return f"File {filename} deleted successfully"

# Creating tasks in a backlog task db
def get_response_with_tools(prompt):
    response = ollama.chat(model='llama3.2',
                           messages=[{'role': 'user', 'content': prompt}],
                           tools=tools.tools)
    # Process tool calls if present
    if 'tool_calls' in response['message']:
        for tool_call in response['message']['tool_calls']:
            if tool_call['function']['name'] == 'add_task':
                task_description = tool_call['function']['arguments']['task_description']
                add_task(task_description)
                print(f"Task added: {task_description}")
            elif tool_call['function']['name'] == 'create_file':
                print("Creating file...")
                filename = tool_call['function']['arguments']['filename']
                content = tool_call['function']['arguments']['content']
                create_file(filename, content)
                print(f"File created: {filename}")
            elif tool_call['function']['name'] == 'read_file':
                print("Reading file...")
                filename = tool_call['function']['arguments']['filename']
                print(f"file name problem {filename}")
                content = read_file(filename)
                print(f"File content: {content}")
            elif tool_call['function']['name'] == 'delete_file':
                print("Deleting file...")
                filename = tool_call['function']['arguments']['filename']
                delete_file(filename)
                print(f"File deleted: {filename}")
    else:
        return response['message']['content']
#get_response_with_tools("Create a task to create a local voice AI assiatant")
#get_response_with_tools("Create a task for Creating a file called 'test.txt' with the content 'Hello, World! and do the task")
record_audio(duration=5)
prompt = transcribe('./prompt.wav')
get_response_with_tools(prompt)
print(tasks_df)
