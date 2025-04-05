from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
import logging
import os
import wave
import io
import json

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LLM_SERVER = "http://127.0.0.1:11434"
DECODER_SERVER = "http://127.0.0.1:11435/embeddings"
API_KEY = "mysecretkey"  # Replace with actual key from 11434
CA_CERT_PATH = "/etc/ssl/certs/kamala_ca.crt"

if not os.path.exists(CA_CERT_PATH):
    logger.warning(
        f"CA certificate not found at {CA_CERT_PATH}. Using system CA bundle."
    )


# Helper functions (unchanged from your last version)
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
    n_fft = 1280
    n_hop = 320
    n_win = 1280
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
    text = re.sub(r"\d+(\.\d+)?", lambda x: x.group(), text.lower())
    text = re.sub(r"[-_/,\.\\]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


@app.post("/audio/speech")
async def generate_speech(request: Request):
    try:
        payload = await request.json()
        text = payload.get("input", "")
        voice_file = payload.get("voice", None)  # Optional speaker file path
        if not text:
            logger.error("No input provided")
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        logger.debug(f"Processing text: {text}")
        words = process_text(text)
        prompt_base = (
            "<|im_start|>\n<|text_start|>" + " ".join(words) + "<|audio_start|>\n"
        )

        # Load speaker profile if provided
        audio_text = ""
        audio_data = ""
        if voice_file:
            try:
                with open(voice_file, "r") as f:
                    speaker = json.load(f)
                audio_text = (
                    "<|text_start|>"
                    + " ".join(word["word"] for word in speaker["words"])
                    + " "
                )
                audio_data = "<|audio_start|>\n"
                for word in speaker["words"]:
                    word_text = word["word"]
                    duration = word["duration"]
                    codes = word["codes"]
                    audio_data += f"{word_text}<|t_{duration:.2f}>"
                    for code in codes:
                        audio_data += f"<|{code}|>"
                    audio_data += "\n"
            except Exception as e:
                logger.error(f"Failed to load voice file: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Voice file error: {str(e)}"
                )

        prompt = audio_text + prompt_base + audio_data

        # LLM request with aiohttp
        logger.debug("Calling LLM server")
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=60)  # 60-second total timeout
            async with session.post(
                f"{LLM_SERVER}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 1024,
                    "cache_prompt": True,
                    "return_tokens": True,
                    "samplers": ["top_k"],
                    "top_k": 16,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "seed": 1003,
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
                ssl=CA_CERT_PATH if os.path.exists(CA_CERT_PATH) else True,
                timeout=timeout,
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
                logger.debug(f"Generated codes: {len(codes)}, codes: {codes[:10]}...")

        # Batch processing with aiohttp
        batch_size = 256
        all_audio = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(codes), batch_size):
                batch = codes[i : i + batch_size]
                logger.debug(
                    f"Processing batch {i//batch_size + 1}: {len(batch)} codes: {batch[:10]}..."
                )
                async with session.post(
                    DECODER_SERVER,
                    json={"input": batch},
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    ssl=CA_CERT_PATH if os.path.exists(CA_CERT_PATH) else True,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    dec_json = await resp.json()
                    logger.debug(f"Decoder response: {dec_json[:100]}...")

                    if (
                        isinstance(dec_json, list)
                        and len(dec_json) > 0
                        and "embedding" in dec_json[0]
                    ):
                        embd = dec_json[0]["embedding"]
                    elif isinstance(dec_json, dict) and "embedding" in dec_json:
                        embd = dec_json["embedding"]
                    else:
                        logger.error(f"Invalid decoder response: {dec_json}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Invalid decoder response: {dec_json}",
                        )

                    audio = embd_to_audio(embd, len(embd), len(embd[0]))
                    audio_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                    all_audio.append(audio_data)

        combined_audio = np.concatenate(all_audio)
        logger.debug(f"Total audio samples: {len(combined_audio)}")

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

    uvicorn.run(app, host="127.0.0.1", port=11436, log_level="debug")
