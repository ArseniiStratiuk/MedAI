"""Record your voice and send it to the MedASR transcription endpoint.

Usage:
    python scripts/test_voice_transcribe.py

Press ENTER to start recording, speak, then press ENTER again to stop.
The audio is sent to POST /api/v1/transcribe and the transcription is printed.

The script records at 48 kHz (native for most mics) and sends the raw
high-quality WAV.  The backend preprocessing pipeline handles resampling,
noise reduction, and normalization before forwarding to MedASR.
"""

import base64
import io
import sys

import httpx
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

API_BASE = "http://localhost:8000/api/v1"
RECORD_RATE = 48000  # Record at native mic rate for best quality


def register_or_login() -> str:
    """Get a JWT token — register a test user or login if exists."""
    creds = {"email": "voice-test@medai.com", "password": "testpass123", "name": "Voice Tester"}

    # Try register first
    r = httpx.post(f"{API_BASE}/auth/register", json=creds, timeout=10)
    if r.status_code == 200:
        return r.json()["access_token"]

    # Already exists — login
    r = httpx.post(
        f"{API_BASE}/auth/login",
        json={"email": creds["email"], "password": creds["password"]},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def record_audio() -> np.ndarray:
    """Record audio from the microphone until the user presses ENTER.

    Records at 48 kHz mono for the best raw quality from typical
    microphones.  The backend handles down-sampling to 16 kHz.
    """
    print("\n🎤  Press ENTER to START recording...")
    input()
    print("🔴  Recording... Speak now! Press ENTER to STOP.\n")

    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  ⚠️  sounddevice: {status}")
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=RECORD_RATE,
        channels=1,
        dtype="float32",
        blocksize=4096,
        callback=callback,
    )
    stream.start()
    input()  # Wait for ENTER to stop
    stream.stop()
    stream.close()

    if not frames:
        print("❌  No audio recorded.")
        sys.exit(1)

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / RECORD_RATE

    # Quick check — warn if too quiet
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"✅  Recorded {duration:.1f}s | RMS={rms:.4f} peak={peak:.4f}")
    if rms < 0.005:
        print("  ⚠️  Audio is very quiet — try speaking louder or moving closer to the mic")

    return audio


def audio_to_base64(audio: np.ndarray) -> str:
    """Convert float32 numpy audio to base64-encoded WAV at the recording rate."""
    # Convert to int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav.write(buf, RECORD_RATE, audio_int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def transcribe(token: str, audio_b64: str) -> dict:
    """Call the /transcribe endpoint."""
    print("📡  Sending to MedASR for transcription...")
    r = httpx.post(
        f"{API_BASE}/transcribe",
        json={"audio_base64": audio_b64},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main():
    print("=" * 50)
    print("  MedAI Voice Transcription Test")
    print("=" * 50)

    # 1. Auth
    print("\n🔐  Authenticating...")
    token = register_or_login()
    print("✅  Authenticated")

    # 2. Record
    audio = record_audio()

    # 3. Encode
    audio_b64 = audio_to_base64(audio)
    print(f"📦  Encoded audio: {len(audio_b64)} chars base64")

    # 4. Transcribe
    result = transcribe(token, audio_b64)

    # 5. Display result
    print("\n" + "=" * 50)
    print("  TRANSCRIPTION RESULT")
    print("=" * 50)
    if result.get("error"):
        print(f"❌  Error: {result['error']}")
    elif result.get("warning"):
        print(f"⚠️  Warning: {result['warning']}")

    transcription = result.get("transcription", "")
    if transcription:
        print(f"\n📝  \"{transcription}\"\n")
        print(f"⏱️  Audio duration: {result.get('duration_seconds', 0):.1f}s")
    else:
        print("\n📝  (empty transcription — try speaking louder or longer)")

    print()


if __name__ == "__main__":
    main()
