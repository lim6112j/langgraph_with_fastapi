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


record_audio()
prompt = transcribe("./prompt.wav")
get_response(prompt)
