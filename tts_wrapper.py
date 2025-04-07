import numpy as np
import aiohttp
import re
import logging
import os
import wave
import io
import ssl
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Logging setup
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
app = FastAPI()

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# Server config
TTSW_HOST = os.getenv("TTSW_HOST", "127.0.0.1")
TTSW_PORT = int(os.getenv("TTSW_PORT", "11436"))
TTSW_AUDIO_INFERENCE_ENDPOINT = os.getenv(
    "TTSW_AUDIO_INFERENCE_ENDPOINT", "http://127.0.0.1:8020/completion"
)
TTSW_AUDIO_DECODER_ENDPOINT = os.getenv(
    "TTSW_AUDIO_DECODER_ENDPOINT", "http://127.0.0.1:8021/embeddings"
)
TTSW_API_KEY = os.getenv("TTSW_API_KEY", None)
TTSW_AUDIO_DECODER_API_KEY = os.getenv("TTSW_AUDIO_DECODER_API_KEY", TTSW_API_KEY)
TTSW_AUDIO_INFERENCE_API_KEY = os.getenv("TTSW_AUDIO_INFERENCE_API_KEY", TTSW_API_KEY)
TTSW_VOICES_DIR = os.getenv("TTSW_VOICES_DIR", "./voices")
TTSW_DISABLE_VERIFY_SSL = (
    os.getenv("TTSW_DISABLE_VERIFY_SSL", "false").lower() == "true"
)

# Inference params
nPredict = int(os.getenv("TTSW_N_PREDICT", "256"))
topK = int(os.getenv("TTSW_TOP_K", "20"))
temperature = float(os.getenv("TTSW_TEMPERATURE", "0.5"))
topP = float(os.getenv("TTSW_TOP_P", "0.5"))
seed = int(os.getenv("TTSW_SEED", "69"))
nFft = int(os.getenv("TTSW_N_FFT", "1280"))
nHop = int(os.getenv("TTSW_N_HOP", "320"))
nWin = int(os.getenv("TTSW_N_WIN", "1280"))

# SSL setup (assuming TTSW_CA_CERT_PATH is defined or removed if unused)
ssl_context = None
if not TTSW_DISABLE_VERIFY_SSL and os.path.exists(
    "/path/to/certs/certfile.crt"
):  # Adjust path if needed
    ssl_context = ssl.create_default_context(cafile="/path/to/certs/certfile.crt")


# Helper functions (unchanged from previous, assuming they’re in the file)
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
    text = re.sub(r"[-_/,\.\\]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if segmentation == "punctuation":
        segments = re.split(r"([.!?])\s+", text)
        segments = [s.strip() for s in segments if s.strip()]
        result = []
        current_segment = []
        for i, seg in enumerate(segments):
            if seg in ".!?":
                if current_segment:
                    result.append(current_segment)
                current_segment = []
            else:
                current_segment.extend(seg.split())
        if current_segment:
            result.append(current_segment)
        return result
    else:
        return [text.split()]


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
        n_predict = min(req.n_predict if req.n_predict is not None else nPredict, 4096)
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'input' in payload")

        logger.info(
            f"Processing text: {text}, segmentation: {segmentation}, n_predict: {n_predict}"
        )

        voice_codes = {}
        if voice_file:
            voice_path = os.path.join(TTSW_VOICES_DIR, f"{voice_file}.json")
            if os.path.exists(voice_path):
                with open(voice_path, "r") as f:
                    voice_data = json.load(f)
                    voice_prefix = voice_data.get("text", "")
                    text = voice_prefix + "\n" + text
                    for word_data in voice_data.get("words", []):
                        voice_codes[word_data["word"]] = word_data["codes"]
            else:
                logger.warning(f"Voice file not found: {voice_path}")

        segments = process_text(text, segmentation)
        logger.debug(f"Processed segments: {segments}")

        all_audio = []
        async with aiohttp.ClientSession() as session:
            for i, segment in enumerate(segments):
                if not segment:
                    continue
                prompt = (
                    "<|im_start|>\n<|text_start|>"
                    + "<|text_sep|>".join(segment)
                    + "<|text_end|>\n"
                )
                logger.debug(f"Processing segment {i+1}/{len(segments)}: {prompt}")

                tts_payload = {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "top_k": topK,
                    "top_p": topP,
                    "seed": seed,
                    "cache_prompt": True,
                    "return_tokens": True,
                    "samplers": ["top_k"],
                }
                for attempt in range(3):
                    try:
                        async with session.post(
                            TTSW_AUDIO_INFERENCE_ENDPOINT,
                            json=tts_payload,
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
                            logger.debug(f"LLM response: {llm_json}")
                            codes = llm_json.get("tokens", [])
                            logger.debug(
                                f"Raw tokens: {codes[:20]}... (total {len(codes)})"
                            )
                            if not codes:
                                logger.warning(
                                    "No tokens in LLM response, using fallback"
                                )
                                codes = [0] * 50
                            else:
                                # Uncomment to test hardcoded codes
                                # codes = [391, 1319, 1478, 895, 1580, 533, 166, 1015, 1169, 1186, 380]  # "so"
                                all_tokens = codes
                                codes = [
                                    t - 151672
                                    for t in codes
                                    if t >= 151672 and t <= 155772
                                ]
                                if not codes or len(codes) < 50:
                                    logger.warning(
                                        "Few/no valid codes in range 151672-155772"
                                    )
                                    logger.debug(
                                        f"Token stats - min: {min(all_tokens)}, max: {max(all_tokens)}"
                                    )
                                    if (
                                        max(all_tokens) < 151672
                                        or min(all_tokens) > 155772
                                    ):
                                        codes = [t for t in all_tokens]
                                        logger.debug(
                                            "Using all tokens due to range mismatch"
                                        )
                            logger.debug(
                                f"Extracted codes: {codes[:10]}... (total {len(codes)})"
                            )
                            break
                    except asyncio.TimeoutError:
                        logger.warning(f"Inference timeout on attempt {attempt + 1}")
                        if attempt == 2:
                            raise HTTPException(
                                status_code=504, detail="Inference timeout"
                            )
                        await asyncio.sleep(5)

                chunk_size = 128
                embeddings = []
                for chunk_start in range(0, len(codes), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(codes))
                    code_chunk = codes[chunk_start:chunk_end]
                    logger.debug(
                        f"Processing chunk {chunk_start}-{chunk_end}: {code_chunk[:10]}..."
                    )

                    for attempt in range(3):
                        try:
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
                                chunk_emb = (
                                    dec_json[0]["embedding"]
                                    if isinstance(dec_json, list)
                                    else dec_json.get("embedding", [])
                                )
                                embeddings.extend(chunk_emb)
                                logger.debug(
                                    f"Embeddings chunk (first 5): {chunk_emb[:5]}"
                                )
                                break
                        except asyncio.TimeoutError:
                            logger.warning(f"Decoder timeout on attempt {attempt + 1}")
                            if attempt == 2:
                                raise HTTPException(
                                    status_code=504, detail="Decoder timeout"
                                )
                            await asyncio.sleep(5)

                if not embeddings:
                    logger.warning(f"No embeddings for segment {i+1}, skipping")
                    continue

                logger.debug(
                    f"Total embeddings: {len(embeddings)} frames, dim: {len(embeddings[0])}"
                )
                audio = embd_to_audio(embeddings, len(embeddings), len(embeddings[0]))
                logger.debug(
                    f"Raw audio: {audio[:10]}, length: {len(audio)}, min/max: {np.min(audio)}, {np.max(audio)}"
                )
                max_abs = max(np.max(np.abs(audio)), 0.01)
                if max_abs > 1.0:
                    audio = audio / max_abs
                logger.debug(f"Normalized audio: {audio[:10]}")
                # Increase fade to 100ms
                fade_samples = 24000 // 10  # 100ms
                if len(audio) > fade_samples * 2:
                    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                audio_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                all_audio.append(audio_data)

        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio generated")

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
