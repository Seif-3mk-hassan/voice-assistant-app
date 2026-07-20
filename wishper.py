# wishper.py
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
import torch
from silero_vad import load_silero_vad

sample_rate = 16000
frames_per_block = 512
block_duration = frames_per_block / sample_rate
channels = 1
silence_timeout = 0.8
min_speech_duration = 0.3
silence_frame_limit = int(silence_timeout / block_duration)

class TTS_class:
    def __init__(self):
        self.vad_model = load_silero_vad()
        self.whisper_model = WhisperModel("medium", device="cuda", compute_type="float32")

    def is_speech(self, audio_chunk):
        tensor = torch.from_numpy(audio_chunk.flatten())
        with torch.no_grad():
            prob = self.vad_model(tensor, sample_rate).item()
        return prob

    def transcribe(self, audio_data):
        audio_data = audio_data.flatten().astype(np.float32)
        segments, info = self.whisper_model.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def listen_and_transcribe(self):
        """Blocks until one utterance is detected, returns the transcribed text."""
        speech_buffer = []
        silence_frames = 0
        in_speech = False

        with sd.InputStream(samplerate=sample_rate, channels=channels, blocksize=frames_per_block) as stream:
            while True:
                block, overflowed = stream.read(frames_per_block)
                prob = self.is_speech(block)

                if prob > 0.5:
                    speech_buffer.append(block)
                    silence_frames = 0
                    in_speech = True
                else:
                    if in_speech:
                        silence_frames += 1
                        speech_buffer.append(block)
                        if silence_frames >= silence_frame_limit:
                            total_duration = len(speech_buffer) * block_duration
                            if total_duration >= min_speech_duration:
                                audio_data = np.concatenate(speech_buffer)
                                return self.transcribe(audio_data)
                            speech_buffer = []
                            silence_frames = 0
                            in_speech = False