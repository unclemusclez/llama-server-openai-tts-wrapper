import aiohttp
import re
import logging
import os
import wave
import io
import json
import ssl
import numpy as np
import time
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Server config
TTSW_HOST = os.getenv("TTSW_HOST", "127.0.0.1")
TTSW_PORT = int(os.getenv("TTSW_PORT", "11436"))
TTSW_AUDIO_INFERENCE_ENDPOINT = os.getenv(
    "TTSW_AUDIO_INFERENCE_ENDPOINT", "http://127.0.0.1:11434/completion"
)
TTSW_AUDIO_DECODER_ENDPOINT = os.getenv(
    "TTSW_AUDIO_DECODER_ENDPOINT", "http://127.0.0.1:11435/embedding"
)
TTSW_API_KEY = os.getenv("TTSW_API_KEY", None)
TTSW_AUDIO_DECODER_API_KEY = os.getenv("TTSW_AUDIO_DECODER_API_KEY", TTSW_API_KEY)
TTSW_AUDIO_INFERENNCE_API_KEY = os.getenv("TTSW_AUDIO_INFERENNCE_API_KEY", TTSW_API_KEY)
TTSW_CA_CERT_PATH = os.getenv("TTSW_CA_CERT_PATH", "/path/to/certs/certfile.crt")
TTSW_VOICES_DIR = os.getenv("TTSW_VOICES_DIR", "./voices")
TTSW_DISABLE_VERIFY_SSL = (
    os.getenv("TTSW_DISABLE_VERIFY_SSL", "false").lower() == "true"
)

# Inference params
nPredict = int(os.getenv("TTSW_N_PREDICT", "1024"))
topK = int(os.getenv("TTSW_TOP_K", "10"))
temperature = float(os.getenv("TTSW_TEMPERATURE", "0.1"))
topP = float(os.getenv("TTSW_TOP_P", "0.1"))
seed = int(os.getenv("TTSW_SEED", "69"))
nFft = int(os.getenv("TTSW_N_FFT", "1280"))
nHop = int(os.getenv("TTSW_N_HOP", "320"))
nWin = int(os.getenv("TTSW_N_WIN", "1280"))
batchSize = int(os.getenv("TTSW_BATCH_SIZE", "256"))

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
        result[start:end] += buffer[i * n_win : (i + 1) * n_win]  # Fixed
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
            S[l, k] = mag * np.exp(1j * phi)  # Fixed

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


def process_text(text: str, segmentation: str = "none"):
    text = text.lower()
    if segmentation == "punctuation":
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [re.sub(r"[^a-z\s]", "", s).split() for s in sentences if s]
    elif segmentation == "paragraph":
        paragraphs = text.split("\n")
        return [re.sub(r"[^a-z\s]", "", p).split() for p in paragraphs if p]
    else:  # "none"
        text = re.sub(r"[-_/,\.\\]", " ", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return [text.split()]


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    segmentation: str | None = "none"


@app.post("/audio/speech")
async def generate_speech(request: Request):
    try:
        payload = await request.json()
        req = SpeechRequest(**payload)
        text, voice_file, segmentation = req.input, req.voice, req.segmentation
        if not text:
            logger.error("No input provided")
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        logger.debug(f"Processing text: {text} with segmentation: {segmentation}")
        segments = process_text(text, segmentation)

        # Load speaker profile if provided
        audio_text, audio_data = "", ""
        if voice_file:
            try:
                voice_path = (
                    voice_file
                    if os.path.isabs(voice_file)
                    else os.path.join(TTSW_VOICES_DIR, voice_file)
                )
                with open(voice_path, "r") as f:
                    speaker = json.load(f)
                audio_text = (
                    "<|text_start|>"
                    + " ".join(word["word"] for word in speaker["words"])
                    + " "
                )
                audio_data = "<|audio_start|>\n"
                for word in speaker["words"]:
                    word_text, duration, codes = (
                        word["word"],
                        word["duration"],
                        word["codes"],
                    )
                    audio_data += f"{word_text}<|t_{duration:.2f}>"
                    for code in codes:
                        audio_data += f"<|{code}|>"
                    audio_data += "\n"
            except Exception as e:
                logger.error(f"Failed to load voice file: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Voice file error: {str(e)}"
                )

        all_audio = []
        async with aiohttp.ClientSession() as session:
            for segment in segments:
                prompt_base = (
                    "<|im_start|>\n<|text_start|>"
                    + " ".join(segment)
                    + "<|audio_start|>\n"
                )
                prompt = audio_text + prompt_base + audio_data

                # LLM request
                logger.debug("Calling LLM server")
                async with session.post(
                    TTSW_AUDIO_INFERENCE_ENDPOINT,
                    json={
                        "prompt": prompt,
                        "n_predict": nPredict,
                        "cache_prompt": True,
                        "return_tokens": True,
                        "samplers": ["top_k"],
                        "top_k": topK,
                        "temperature": temperature,
                        "top_p": topP,
                        "seed": seed,
                    },
                    headers={"Authorization": f"Bearer {TTSW_AUDIO_INFERENCE_API_KEY}"},
                    ssl=(
                        False
                        if TTSW_AUDIO_INFERENCE_ENDPOINT.startswith("http://")
                        else ssl_context
                    ),
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    llm_json = await resp.json()
                    if "tokens" not in llm_json:
                        logger.error(f"LLM response missing 'tokens': {llm_json}")
                        raise HTTPException(
                            status_code=500, detail="LLM server did not return tokens"
                        )
                    codes = [
                        t - 151672 for t in llm_json["tokens"] if 151672 <= t <= 155772
                    ]
                    logger.debug(
                        f"Generated codes: {len(codes)}, codes: {codes[:10]}..."
                    )

                # Batch processing with timing
                for i in range(0, len(codes), batchSize):
                    batch = codes[i : i + batchSize]
                    logger.debug(
                        f"Processing batch {i//batchSize + 1}: {len(batch)} codes: {batch[:10]}..."
                    )
                    start_time = time.time()
                    async with session.post(
                        TTSW_AUDIO_DECODER_ENDPOINT,
                        json={"input": batch},
                        headers={
                            "Authorization": f"Bearer {TTSW_AUDIO_DECODER_API_KEY}"
                        },
                        ssl=(
                            False
                            if TTSW_AUDIO_DECODER_ENDPOINT.startswith("http://")
                            else ssl_context
                        ),
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        resp.raise_for_status()
                        dec_json = await resp.json()
                        embd = dec_json  # Directly use the list response
                        if not embd or not isinstance(embd, list):
                            logger.error(f"Invalid decoder response: {dec_json}")
                            raise HTTPException(
                                status_code=500, detail="Invalid decoder response"
                            )
                        audio = embd_to_audio(embd, len(embd), len(embd[0]))
                        audio_data = np.clip(audio * 32767, -32768, 32767).astype(
                            np.int16
                        )
                        all_audio.append(audio_data)
                        logger.debug(
                            f"Batch {i//batchSize + 1} took {time.time() - start_time:.2f}s"
                        )
        # Concatenate all audio into a single array
        combined_audio = (
            np.concatenate(all_audio) if all_audio else np.array([], dtype=np.int16)
        )
        logger.debug(f"Total audio samples: {len(combined_audio)}")

        # Write to WAV
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.setnframes(len(combined_audio))
            wav_file.writeframes(combined_audio.tobytes())

        wav_data = wav_io.getvalue()
        logger.debug(f"Generated WAV: {len(wav_data)} bytes")
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={"Content-Length": str(len(wav_data))},
        )

    except aiohttp.ClientError as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS server error: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal TTS error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=TTSW_HOST, port=TTSW_PORT, log_level="debug")
