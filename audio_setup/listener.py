"""
Live transcription using openai-whisper and sounddevice.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import whisper
import sys
import time
from datetime import datetime

# VAD tunables
SILENCE_THRESHOLD   = 0.01
SPEECH_THRESHOLD    = 0.02
SILENCE_TIMEOUT_SEC = 1.5

audio_queue: queue.Queue = queue.Queue()
stop_event   = threading.Event()
transcription_result = ""


def list_devices():
    """Print available audio input devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print()


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for every audio block."""
    if status:
        print(f"[sounddevice status] {status}", file=sys.stderr)
    # indata shape: (frames, channels) — we keep a copy
    audio_queue.put(indata.copy())


def record_worker(device_index, sample_rate, channels, chunk_seconds):
    """Opens the input stream and feeds audio into audio_queue."""
    chunk_frames = int(sample_rate * chunk_seconds)

    # print(chunk_seconds)

    # print(f"[info] Opening device {device_index} at {sample_rate} Hz …")
    try:
        with sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=chunk_frames,
            callback=audio_callback,
        ):
            print("[info] Recording started. Press Ctrl+C to stop.")
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as exc:
        print(f"[error] Could not open device {device_index}: {exc}", file=sys.stderr)
        stop_event.set()


def transcribe_worker(model=whisper.Whisper):
    global transcription_result
    speech_detected   = False
    last_speech_time  = None
    accumulated_audio = []

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        audio_np = audio_chunk.flatten().astype(np.float32)
        rms       = float(np.sqrt(np.mean(audio_np ** 2)))
        is_speech = rms >= SPEECH_THRESHOLD

        if not speech_detected:
            if is_speech:
                # print("[info] Speech detected — listening …")
                speech_detected  = True
                last_speech_time = time.monotonic()
                accumulated_audio.append(audio_np)
            continue

        accumulated_audio.append(audio_np)

        if is_speech:
            last_speech_time = time.monotonic()
        else:
            silence_duration = time.monotonic() - last_speech_time
            if silence_duration >= SILENCE_TIMEOUT_SEC:
                stop_event.set()
                transcription_result = _transcribe_utterance(model, accumulated_audio)
                # ts = datetime.now().strftime("%H:%M:%S")
                # print(f"[{ts}] {transcription_result}")

    print(f"[info] Transcription stopped, captured: {transcription_result}")


def _transcribe_utterance(model: whisper.Whisper, chunks: list[np.ndarray]) -> str:
    """Concatenate buffered chunks, run Whisper, and return the transcript."""
    audio_np = np.concatenate(chunks)

    if np.abs(audio_np).max() < SILENCE_THRESHOLD:
        return ""

    result = model.transcribe(
        audio_np,
        fp16=False,
        language="en",
    )
    return result["text"].strip()