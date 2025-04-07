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
    # Always process as a single string, no splitting
    text = text.lower()
    text = re.sub(r"[-_/,\.\\]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        text, voice_file = req.input, req.voice
        n_predict = min(req.n_predict if req.n_predict is not None else nPredict, 4096)
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        logger.info(f"Processing text: {text}, n_predict: {n_predict}")

        # Load voice data
        voice_data = None
        default_voice_path = os.path.join(TTSW_VOICES_DIR, "en_female_1.json")

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
                logger.debug(f"Loaded specified voice file: {voice_path}")
            else:
                logger.warning(f"Specified voice file not found: {voice_path}")

        if not voice_data:
            if os.path.exists(default_voice_path):
                with open(default_voice_path, "r") as f:
                    voice_data = json.load(f)
                logger.debug(f"Using default voice: {default_voice_path}")
            else:
                logger.warning(f"Default voice file not found: {default_voice_path}")

        if not voice_data:
            logger.warning(
                "No voice data available; proceeding with default model behavior"
            )

        # Process text as a single string
        processed_text = process_text(text)
        logger.debug(f"Processed text: {processed_text}")

        all_audio = []
        async with aiohttp.ClientSession() as session:
            prompt = f"<|im_start|>\n<|text_start|>{processed_text}<|text_end|>\n<|audio_start|>\n"
            logger.debug(f"Processing prompt: {prompt}")
            minimal_payload = {
                "prompt": [prompt],
                "n_predict": n_predict,
                "cache_prompt": True,
                "return_tokens": True,
                "samplers": ["top_k", "temperature", "top_p"],
                "top_k": topK,
                "temperature": temperature,
                "top_p": topP,
                "seed": seed,
            }
            if voice_data:
                minimal_payload["voice"] = voice_data
            logger.debug(f"Minimal payload: {minimal_payload}")

            # Inference request
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
                            logger.error(f"LLM response missing 'tokens': {llm_json}")
                            raise HTTPException(
                                status_code=500,
                                detail="LLM server did not return tokens",
                            )
                        codes = [
                            t - 151672
                            for t in llm_json["tokens"]
                            if 151672 <= t <= 155772
                        ]
                        if not codes or len(codes) < 50:
                            logger.warning(
                                f"Few/no valid codes in range 151672-155772."
                            )
                            all_tokens = llm_json["tokens"]
                            if max(all_tokens) < 151672 or min(all_tokens) > 155772:
                                codes = [t for t in all_tokens]
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
                logger.warning("No valid codes generated. Adding silence.")
                all_audio.append(np.zeros(24000 // 2, dtype=np.int16))
            else:
                chunk_size = batchSize
                embeddings = []
                for chunk_start in range(0, len(codes), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(codes))
                    code_chunk = codes[chunk_start:chunk_end]
                    decoder_payload = {"input": code_chunk}
                    if voice_data:
                        decoder_payload["voice"] = voice_data
                    logger.debug(f"Decoder payload: {decoder_payload}")

                    for attempt in range(3):
                        try:
                            async with session.post(
                                TTSW_AUDIO_DECODER_ENDPOINT,
                                json=decoder_payload,
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
                                        f"Invalid decoder response: {dec_json}"
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail="Invalid decoder response",
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
                audio = embd_to_audio(embd, len(embd), len(embd[0]))
                max_abs = np.max(np.abs(audio))
                if max_abs > 1.0:
                    audio = audio / max_abs
                audio_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                all_audio.append(audio_data)

        combined_audio = np.concatenate(all_audio)
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
