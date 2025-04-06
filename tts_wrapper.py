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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tts_wrapper.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
app = FastAPI()
logger.debug("TTS wrapper initialized")

# Enable aiohttp logging
aiohttp_logger = logging.getLogger("aiohttp.client")
aiohttp_logger.setLevel(logging.DEBUG)
aiohttp_logger.propagate = True

# Load environment variables
load_dotenv()

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
topK = int(os.getenv("TTSW_TOP_K", "10"))
temperature = float(os.getenv("TTSW_TEMPERATURE", "0.1"))
topP = float(os.getenv("TTSW_TOP_P", "0.1"))
seed = int(os.getenv("TTSW_SEED", "69"))
nFft = int(os.getenv("TTSW_N_FFT", "1280"))
nHop = int(os.getenv("TTSW_N_HOP", "320"))
nWin = int(os.getenv("TTSW_N_WIN", "1280"))

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
        return text.split()  # Return flat list, not wrapped in [list]


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    segmentation: str | None = "none"
    n_predict: int | None = None


@app.post("/audio/speech")
async def generate_speech(request: Request):
    try:
        payload = await request.json()
        req = SpeechRequest(**payload)
        text, voice_file, segmentation = req.input, req.voice, req.segmentation
        n_predict = req.n_predict if req.n_predict is not None else nPredict
        n_predict = min(n_predict, 4096)
        if not text:
            logger.error("No input provided")
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        logger.info(
            f"Processing text: {text} with segmentation: {segmentation}, n_predict: {n_predict}"
        )

        voice_data = None
        if voice_file:
            voice_path = os.path.join(
                TTSW_VOICES_DIR,
                (
                    f"{voice_file}.json"
                    if not voice_file.endswith(".json")
                    else voice_file
                ),
            )
            if os.path.exists(voice_path):
                with open(voice_path, "r") as f:
                    voice_data = json.load(f)
                logger.debug(f"Loaded voice file: {voice_path} with data: {voice_data}")
            else:
                logger.warning(f"Voice file not found: {voice_path}")
                raise HTTPException(
                    status_code=400, detail=f"Voice file '{voice_file}' not found"
                )

        segments = process_text(text, segmentation="punctuation")
        logger.debug(f"Processed segments: {segments}")

        all_audio = []
        async with aiohttp.ClientSession() as session:
            for i, segment in enumerate(segments):
                if not segment:
                    continue
                prompt = (
                    "<|im_start|>\n<|text_start|>"
                    + "<|text_sep|>".join(segment)
                    + "<|text_end|>\n<|audio_start|>\n"
                )
                logger.debug(f"Processing segment {i+1}/{len(segments)}: {prompt}")

                minimal_payload = {
                    "prompt": [prompt],
                    "n_predict": n_predict,
                    "cache_prompt": True,
                    "return_tokens": True,
                    "samplers": ["top_k", "temperature", "top_p"],
                    "top_k": 50,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "seed": 1003,
                }
                if voice_data:
                    minimal_payload["voice"] = voice_data
                logger.debug(f"Minimal payload: {minimal_payload}")
                for attempt in range(3):
                    try:
                        async with session.post(
                            TTSW_AUDIO_INFERENCE_ENDPOINT,
                            json=minimal_payload,
                            headers={
                                "Authorization": f"Bearer {TTSW_AUDIO_INFERENCE_API_KEY}"
                            },
                            ssl=(
                                False
                                if TTSW_AUDIO_INFERENCE_ENDPOINT.startswith("http://")
                                else ssl_context
                            ),
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            resp.raise_for_status()
                            llm_json = await resp.json()
                            logger.debug(f"Inference response: {llm_json}")
                            if "tokens" not in llm_json:
                                logger.error(
                                    f"LLM response missing 'tokens': {llm_json}"
                                )
                                raise HTTPException(
                                    status_code=500,
                                    detail="LLM server did not return tokens",
                                )
                            logger.debug(
                                f"Raw tokens: {llm_json['tokens'][:20]}... (total {len(llm_json['tokens'])})"
                            )
                            codes = [
                                t - 151672
                                for t in llm_json["tokens"]
                                if 151672 <= t <= 155772
                            ]
                            if not codes or len(codes) < 50:
                                logger.warning(
                                    "Few/no valid codes in range 151672-155772."
                                )
                                all_tokens = llm_json["tokens"]
                                logger.debug(
                                    f"Token stats - min: {min(all_tokens)}, max: {max(all_tokens)}"
                                )
                                if max(all_tokens) < 151672 or min(all_tokens) > 155772:
                                    codes = [t for t in all_tokens]
                                    logger.debug(
                                        "Using all tokens as codes due to range mismatch."
                                    )
                            logger.debug(
                                f"Generated codes: {codes[:10]}... (total {len(codes)})"
                            )
                            break
                    except asyncio.TimeoutError:
                        logger.warning(f"Inference timeout on attempt {attempt + 1}")
                        if attempt == 2:
                            raise HTTPException(
                                status_code=504,
                                detail="Inference endpoint timed out after retries",
                            )
                        await asyncio.sleep(5)
                    except aiohttp.ClientError as e:
                        logger.error(f"Inference request failed: {e}", exc_info=True)
                        if attempt == 2:
                            raise HTTPException(
                                status_code=500, detail=f"Inference error: {str(e)}"
                            )
                        await asyncio.sleep(5)

                if not codes:
                    logger.warning(f"No valid codes for segment {i+1}. Skipping.")
                    continue

                logger.debug(f"Processing {len(codes)} codes in chunks...")
                chunk_size = 128  # Match physical batch size
                embeddings = []
                for chunk_start in range(0, len(codes), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(codes))
                    code_chunk = codes[chunk_start:chunk_end]
                    logger.debug(
                        f"Processing chunk {chunk_start}-{chunk_end} of {len(codes)}: {code_chunk[:10]}..."
                    )

                    for attempt in range(3):
                        try:
                            logger.debug(
                                f"Sending decoder request to {TTSW_AUDIO_DECODER_ENDPOINT}"
                            )
                            async with session.post(
                                TTSW_AUDIO_DECODER_ENDPOINT,
                                json={"input": code_chunk},
                                headers={
                                    "Authorization": f"Bearer {TTSW_AUDIO_DECODER_API_KEY}"
                                },
                                ssl=(
                                    False
                                    if TTSW_AUDIO_DECODER_ENDPOINT.startswith("http://")
                                    else ssl_context
                                ),
                                timeout=aiohttp.ClientTimeout(total=30),
                            ) as resp:
                                resp.raise_for_status()
                                dec_json = await resp.json()
                                logger.debug(f"Decoder response for chunk: {dec_json}")
                                if (
                                    isinstance(dec_json, list)
                                    and dec_json
                                    and "embedding" in dec_json[0]
                                ):
                                    embeddings.extend(dec_json[0]["embedding"])
                                elif (
                                    isinstance(dec_json, dict)
                                    and "embedding" in dec_json
                                ):
                                    embeddings.extend(dec_json["embedding"])
                                else:
                                    logger.error(
                                        f"Invalid decoder response for chunk: {dec_json}"
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail="Invalid decoder chunk response",
                                    )
                                break
                        except asyncio.TimeoutError:
                            logger.warning(f"Decoder timeout on attempt {attempt + 1}")
                            if attempt == 2:
                                raise HTTPException(
                                    status_code=504,
                                    detail="Decoder timed out after retries",
                                )
                            await asyncio.sleep(5)
                        except aiohttp.ClientError as e:
                            logger.error(f"Decoder request failed: {e}", exc_info=True)
                            if attempt == 2:
                                raise HTTPException(
                                    status_code=500, detail=f"Decoder error: {str(e)}"
                                )
                            await asyncio.sleep(5)

                embd = embeddings
                logger.debug(
                    f"Embeddings extracted: {embd[:10]}... (total {len(embd)})"
                )
                embd = [[float(x) for x in row if abs(float(x)) < 1e5] for row in embd]
                if not embd or not all(len(row) == len(embd[0]) for row in embd):
                    logger.error(f"Invalid embeddings after sanitization: {embd}")
                    raise HTTPException(
                        status_code=500,
                        detail="Embeddings contain inconsistent or invalid data",
                    )
                logger.debug(
                    f"Sanitized embeddings: {embd[:10]}... (total {len(embd)})"
                )
                if (
                    not embd
                    or not isinstance(embd, list)
                    or not all(isinstance(e, list) for e in embd)
                ):
                    logger.error(f"Invalid embeddings format: {embd}")
                    raise HTTPException(
                        status_code=500, detail="Decoder returned invalid embeddings"
                    )

                logger.debug(
                    f"Synthesis params - n_fft: {nFft}, n_hop: {nHop}, n_win: {nWin}"
                )
                audio = embd_to_audio(embd, len(embd), len(embd[0]))
                logger.debug(f"Audio generated: {len(audio)} samples")
                max_abs = np.max(np.abs(audio))
                if max_abs > 1.0:
                    audio = audio / max_abs
                    logger.debug(
                        f"Normalized audio to avoid clipping. New min/max: {np.min(audio)}, {np.max(audio)}"
                    )
                fade_samples = 24000 // 4
                audio[:fade_samples] = audio[:fade_samples] * np.linspace(
                    0, 1, fade_samples
                )
                audio[-fade_samples:] = audio[-fade_samples:] * np.linspace(
                    1, 0, fade_samples
                )
                logger.debug(f"Audio min/max: {np.min(audio)}, {np.max(audio)}")
                audio_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                all_audio.append(audio_data)

        if not all_audio:
            logger.error("No audio segments generated")
            raise HTTPException(status_code=500, detail="No audio segments generated")

        logger.debug(f"Processed {len(all_audio)} segments out of {len(segments)}")
        combined_audio = np.concatenate(all_audio)
        logger.debug(f"Combined audio: {len(combined_audio)} samples")

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
        logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS server error: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal TTS error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=TTSW_HOST, port=TTSW_PORT, log_level="debug")
