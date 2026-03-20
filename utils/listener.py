#!/usr/bin/env python3
"""
Live transcription using openai-whisper and sounddevice.
Microphone is set to device index 1.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import whisper
import sys
import time
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE_INDEX   = 1          # Your microphone device
SAMPLE_RATE    = 16000      # Whisper expects 16 kHz
CHANNELS       = 1          # Mono
CHUNK_SECONDS  = 5          # How many seconds of audio to transcribe at once
WHISPER_MODEL  = "tiny"     # tiny | base | small | medium | large
# ─────────────────────────────────────────────────────────────────────────────

# ── TRIGGER OPTIONS ──────────────────────────────────────────────────────────
EXIT_OPT       = None
try:
    exit_opt = sys.argv[1]
    if exit_opt == "--trigger":
        EXIT_OPT = "trigger"

    elif exit_opt == "--quiet":
        EXIT_OPT = "quiet"
except Exception:
    pass

START_TRIGGER = "hello"
STOP_TRIGGER = "bye"
exit_stop_trigger = 0
# ─────────────────────────────────────────────────────────────────────────────

audio_queue: queue.Queue = queue.Queue()
stop_event   = threading.Event()

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


def record_worker():
    """Opens the input stream and feeds audio into audio_queue."""
    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)

    # print(f"[info] Opening device {DEVICE_INDEX} at {SAMPLE_RATE} Hz …")
    try:
        with sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_frames,
            callback=audio_callback,
        ):
            # print("[info] Recording started. Press Ctrl+C to stop.\n")
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as exc:
        # print(f"[error] Could not open device {DEVICE_INDEX}: {exc}", file=sys.stderr)
        stop_event.set()


def transcribe_worker(model: whisper.Whisper):
    """Pulls audio chunks from the queue and transcribes them."""
    # print(f"[info] Whisper '{WHISPER_MODEL}' model loaded.\n")
    # print("=" * 60)

    speech_gathered = False
    global exit_stop_trigger

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Flatten to 1-D float32 numpy array (required by Whisper)
        audio_np = audio_chunk.flatten().astype(np.float32)

        # Skip nearly silent chunks to avoid hallucinations
        if np.abs(audio_np).max() < 0.01:
            continue

        result = model.transcribe(
            audio_np,
            fp16=False,       # fp16=True requires a CUDA GPU
            language="en",    # auto-detect; set e.g. "en" to fix language
        )

        text = result["text"].strip()
        if text:
            # set flag to indicate speech identified
            speech_gathered = True
            print(text)

            if STOP_TRIGGER in text.lower():
                exit_stop_trigger = 1
                stop_event.set()

            if EXIT_OPT == "trigger" and START_TRIGGER in text.lower():
                stop_event.set()

        elif EXIT_OPT == "quiet" and speech_gathered:
            stop_event.set()

    # print("=" * 60)
    # print("[info] Transcription stopped.")


def main():
    # list_devices()

    # print(f"[info] Loading Whisper model '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)

    # Start recording thread
    rec_thread = threading.Thread(target=record_worker, daemon=True)
    rec_thread.start()

    trans_thread = threading.Thread(
        target=transcribe_worker, args=(model,), daemon=True
    )

    trans_thread.start()

    try:
        while rec_thread.is_alive() or trans_thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        # print("\n[info] Stopping …")
        stop_event.set()
        rec_thread.join(timeout=3)
        trans_thread.join(timeout=10)

    sys.exit(exit_stop_trigger)


if __name__ == "__main__":
    main()