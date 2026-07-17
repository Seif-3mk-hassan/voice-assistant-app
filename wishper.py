import os

base = r"C:\Users\Seif2\AppData\Local\Programs\Python\Python313\Lib\site-packages\nvidia"
dll_paths = [
    os.path.join(base, "cublas", "bin"),
    os.path.join(base, "cudnn", "bin"),
    os.path.join(base, "cuda_runtime", "bin"),
    os.path.join(base, "nvjitlink", "bin"),
]

os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ["PATH"]

from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
import threading

#settings 
sample_rate = 16000
block_duration = 0.5  # seconds
chunk_duration = 2 # seconds
channels = 1

model_size = "medium"  # or "small", "large-v1", "large-v2"

frames_per_block = int(sample_rate * block_duration)
frames_per_chunk = int(sample_rate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float32")

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback,blocksize=frames_per_block):
        print("Recording audio... Press Ctrl+C to stop.")
        while True:
            sd.sleep(int(block_duration * 1000))

def transcribe_audio():
    global audio_buffer
    while True:
        # Get audio data from the queue
        audio_data = audio_queue.get(timeout=1)
        audio_buffer.append(audio_data)
        
        total_frames = sum(len(buffer) for buffer in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []
            
            audio_data = audio_data.flatten().astype(np.float32)
            segments, info = model.transcribe(audio_data, language="en", beam_size=1, 
                                            no_speech_threshold=0.6,       # skip segments likely to be silence
                                            condition_on_previous_text=False,  # stops it "drifting" based on prior hallucinated text
                                            vad_filter=True,               # built-in VAD to strip silence before transcribing
                            )
            for segment in segments:
                print(segment.text)
                
if __name__ == "__main__":
    threading.Thread(target=record_audio, daemon=True).start()
    transcribe_audio()    