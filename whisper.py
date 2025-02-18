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
result = pipe("./test-audio-file.mp3")
print(result["text"])
