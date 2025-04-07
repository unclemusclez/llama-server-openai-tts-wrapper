import aiohttp
import re
import logging
import os
import wave
import io
import ssl
import numpy as np
import time
import asyncio
import json

from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


# Load logging configuration from file
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
app = FastAPI()
logger.debug("TTS wrapper initialized")

# Enable aiohttp logging
aiohttp_logger = logging.getLogger("aiohttp.client")
aiohttp_logger.setLevel(logging.DEBUG)
aiohttp_logger.propagate = True


# Load environment variables explicitly
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)


# Server config
TTSW_HOST = os.getenv("TTSW_HOST", "127.0.0.1")
TTSW_PORT = int(os.getenv("TTSW_PORT", "11436"))
TTSW_AUDIO_INFERENCE_ENDPOINT = os.getenv(
    "TTSW_AUDIO_INFERENCE_ENDPOINT", "http://127.0.0.1:11434/completion"
)
TTSW_AUDIO_DECODER_ENDPOINT = os.getenv(
    "TTSW_AUDIO_DECODER_ENDPOINT",
    "http://127.0.0.1:11435/embeddings",  # Changed to plural
)
TTSW_API_KEY = os.getenv("TTSW_API_KEY", None)
TTSW_AUDIO_DECODER_API_KEY = os.getenv("TTSW_AUDIO_DECODER_API_KEY", TTSW_API_KEY)
TTSW_AUDIO_INFERENCE_API_KEY = os.getenv("TTSW_AUDIO_INFERENCE_API_KEY", TTSW_API_KEY)
TTSW_CA_CERT_PATH = os.getenv("TTSW_CA_CERT_PATH", "/path/to/certs/certfile.crt")
TTSW_VOICES_DIR = os.getenv("TTSW_VOICES_DIR", "./voices")
TTSW_DISABLE_VERIFY_SSL = (
    os.getenv("TTSW_DISABLE_VERIFY_SSL", "false").lower() == "true"
)
# Inference params
nPredict = int(os.getenv("TTSW_N_PREDICT", "1024"))
batchSize = int(os.getenv("TTSW_BATCH_SIZE", "64"))
topK = int(os.getenv("TTSW_TOP_K", "50"))
temperature = float(os.getenv("TTSW_TEMPERATURE", "0.8"))
topP = float(os.getenv("TTSW_TOP_P", "0.95"))
seed = int(os.getenv("TTSW_SEED", "42"))
nFft = int(os.getenv("TTSW_N_FFT", "1280"))
nHop = int(os.getenv("TTSW_N_HOP", "320"))
nWin = int(os.getenv("TTSW_N_WIN", "1280"))
logger.debug(
    f"Loaded env vars - topK: {topK}, temperature: {temperature}, topP: {topP}, seed: {seed}"
)

if not os.path.exists(TTSW_CA_CERT_PATH):
    logger.warning(
        f"CA certificate not found at {TTSW_CA_CERT_PATH}. Using default SSL."
    )

# SSLContext setup
ssl_context = None
if os.path.exists(TTSW_CA_CERT_PATH) and not TTSW_DISABLE_VERIFY_SSL:
    ssl_context = ssl.create_default_context(cafile=TTSW_CA_CERT_PATH)


# Helper functions
def fill_hann_window(size, periodic=True):
    if periodic:
        return np.hanning(size + 1)[:-1]
    return np.hanning(size)


def irfft(n_fft, complex_input):
    return np.fft.irfft(complex_input, n=n_fft)


def fold(buffer, n_out, n_win, n_hop, n_pad):
    result = np.zeros(n_out)
    n_frames = len(buffer) // n_win
    for i in range(n_frames):
        start = i * n_hop
        end = start + n_win
        result[start:end] += buffer[i * n_win : (i + 1) * n_win]
    return result[n_pad:-n_pad] if n_pad > 0 else result


def process_frame(args):
    l, n_fft, ST, hann = args
    frame = irfft(n_fft, ST[l])
    frame = frame * hann
    hann2 = hann * hann
    return frame, hann2


def embd_to_audio(embd, n_codes, n_embd, n_thread=4):
    embd = np.asarray(embd, dtype=np.float32).reshape(n_codes, n_embd)
    n_fft, n_hop, n_win = nFft, nHop, nWin
    n_pad = (n_win - n_hop) // 2
    n_out = (n_codes - 1) * n_hop + n_win
    hann = fill_hann_window(n_fft, True)

    E = np.zeros((n_embd, n_codes), dtype=np.float32)
    for l in range(n_codes):
        for k in range(n_embd):
            E[k, l] = embd[l, k]

    half_embd = n_embd // 2
    S = np.zeros((n_codes, half_embd + 1), dtype=np.complex64)
    for k in range(half_embd):
        for l in range(n_codes):
            mag = E[k, l]
            phi = E[k + half_embd, l]
            mag = np.clip(np.exp(mag), 0, 1e2)
            S[l, k] = mag * np.exp(1j * phi)

    res = np.zeros(n_codes * n_fft)
    hann2_buffer = np.zeros(n_codes * n_fft)
    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        args = [(l, n_fft, S, hann) for l in range(n_codes)]
        results = list(executor.map(process_frame, args))
        for l, (frame, hann2) in enumerate(results):
            res[l * n_fft : (l + 1) * n_fft] = frame
            hann2_buffer[l * n_fft : (l + 1) * n_fft] = hann2

    audio = fold(res, n_out, n_win, n_hop, n_pad)
    env = fold(hann2_buffer, n_out, n_win, n_hop, n_pad)
    mask = env > 1e-10
    audio[mask] /= env[mask]
    return audio


def process_text(text: str):
    text = text.lower()
    text = re.sub(r"[-_/,\.\\]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    formatted_text = (
        "<|im_start|>\n<|text_start|>" + "<|text_sep|>".join(words) + "<|text_end|>\n"
    )
    return formatted_text


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    segmentation: str | None = "none"
    n_predict: int | None = None


@app.get("/audio/voices")
async def get_available_voices():
    """Return a dictionary of available voices from the voices directory."""
    try:
        voices_dir = TTSW_VOICES_DIR
        if not os.path.exists(voices_dir):
            logger.warning(f"Voices directory not found: {voices_dir}")
            return {}

        available_voices = {}
        for filename in os.listdir(voices_dir):
            if filename.endswith(".json"):
                voice_id = filename[:-5]  # Remove '.json' extension
                voice_path = os.path.join(voices_dir, filename)
                try:
                    with open(voice_path, "r") as f:
                        voice_data = json.load(f)
                        # Use 'name' field if present, otherwise fallback to filename
                        voice_name = voice_data.get("name", voice_id)
                        available_voices[voice_id] = voice_name
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading voice file {voice_path}: {str(e)}")
                    # Fallback to filename if JSON is invalid or unreadable
                    available_voices[voice_id] = voice_id

        logger.debug(f"Available voices: {available_voices}")
        return {"voices": [{"id": k, "name": v} for k, v in available_voices.items()]}

    except Exception as e:
        logger.error(f"Error fetching voices: {str(e)}", exc_info=True)
        return {}


@app.post("/audio/speech")
async def generate_speech(request: Request):
    try:
        payload = await request.json()
        req = SpeechRequest(**payload)
        text, voice_file = req.input, req.voice
        n_predict = min(req.n_predict if req.n_predict is not None else nPredict, 4096)
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        # Load voice profile if provided
        if voice_file:
            voice_path = os.path.join(TTSW_VOICES_DIR, f"{voice_file}.json")
            if os.path.exists(voice_path):
                with open(voice_path, "r") as f:
                    voice_data = json.load(f)
                    voice_prefix = voice_data.get("text", "")
                    text = voice_prefix + "\n" + text

        logger.info(f"Processing text: {text}, n_predict: {n_predict}")
        processed_text = process_text(text)

        async with aiohttp.ClientSession() as session:
            # TTS request to LLM
            tts_payload = {
                "prompt": processed_text,
                "n_predict": n_predict,
                "temperature": temperature,
                "top_k": topK,
                "top_p": topP,
                "seed": seed,
                "return_tokens": True,
                "samplers": ["top_k"],
            }
            async with session.post(
                TTSW_AUDIO_INFERENCE_ENDPOINT,
                json=tts_payload,
                headers={"Authorization": f"Bearer {TTSW_AUDIO_INFERENCE_API_KEY}"},
                ssl=(
                    False
                    if TTSW_AUDIO_INFERENCE_ENDPOINT.startswith("http://")
                    else ssl_context
                ),
            ) as resp:
                resp.raise_for_status()
                llm_json = await resp.json()
                logger.debug(f"LLM response: {llm_json}")
                codes = llm_json.get("tokens", [])
                if not codes:
                    logger.warning("No tokens in LLM response, using fallback")
                    codes = [0] * 50
                else:
                    codes = [t - 151672 for t in codes if t >= 151672 and t <= 155772]
                    if not codes:
                        logger.warning("Filtered codes empty, using fallback")
                        codes = [0] * 50
                logger.debug(f"Extracted codes: {codes[:10]}... (total {len(codes)})")

            # Decoder request
            decoder_payload = {"input": codes}
            async with session.post(
                TTSW_AUDIO_DECODER_ENDPOINT,
                json=decoder_payload,
                headers={"Authorization": f"Bearer {TTSW_AUDIO_DECODER_API_KEY}"},
                ssl=(
                    False
                    if TTSW_AUDIO_DECODER_ENDPOINT.startswith("http://")
                    else ssl_context
                ),
            ) as resp:
                resp.raise_for_status()
                dec_json = await resp.json()
                embeddings = (
                    dec_json.get("embedding", []) if isinstance(dec_json, dict) else []
                )
                if not embeddings and isinstance(dec_json, list) and dec_json:
                    embeddings = dec_json[0].get("embedding", [])
                if not embeddings:
                    raise HTTPException(
                        status_code=500, detail="No embeddings from decoder"
                    )

            # Audio conversion
            audio = embd_to_audio(embeddings, len(embeddings), len(embeddings[0]))
            max_abs = max(np.max(np.abs(audio)), 0.01)
            if max_abs > 1.0:
                audio = audio / max_abs
            audio_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        # Generate WAV
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.setnframes(len(audio_data))
            wav_file.writeframes(audio_data.tobytes())

        wav_data = wav_io.getvalue()
        logger.debug(f"Generated WAV: {len(wav_data)} bytes")
        with open("output.wav", "wb") as f:
            f.write(wav_data)

        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={"Content-Length": str(len(wav_data))},
        )

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal TTS error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=TTSW_HOST, port=TTSW_PORT, log_level="debug")
