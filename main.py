import asyncio
import json
import time
import hashlib
import bisect
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceGateway")

app = FastAPI(title="High-Concurrency Streaming Voice Gateway")

# -----------------------------------------------------------------------------
# 1. KV-Cache Aware Balancer (Consistent Hashing Ring)
# -----------------------------------------------------------------------------
class HashRing:
    def __init__(self, nodes: List[str] = None, replicas: int = 50):
        self.replicas = replicas
        self.ring: List[int] = []
        self.node_map: Dict[int, str] = {}
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) & 0xffffffff

    def add_node(self, node: str):
        for i in range(self.replicas):
            val = self._hash(f"{node}#{i}")
            bisect.insort(self.ring, val)
            self.node_map[val] = node

    def remove_node(self, node: str):
        for i in range(self.replicas):
            val = self._hash(f"{node}#{i}")
            idx = bisect.bisect_left(self.ring, val)
            if idx < len(self.ring) and self.ring[idx] == val:
                self.ring.pop(idx)
                del self.node_map[val]

    def get_node(self, key: str) -> str:
        if not self.ring:
            return ""
        val = self._hash(key)
        idx = bisect.bisect_right(self.ring, val)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]

workers = [
    "GPU-Worker-Node-0 (A100-80GB)",
    "GPU-Worker-Node-1 (A100-80GB)",
    "GPU-Worker-Node-2 (A100-80GB)"
]
balancer = HashRing(workers)

# -----------------------------------------------------------------------------
# 2. Language Detection & Indic Support
# -----------------------------------------------------------------------------
# Unicode block ranges for major Indic scripts
_INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi"),  # Devanagari  → Hindi / Marathi
    (0x0980, 0x09FF, "bn"),  # Bengali     → Bengali
    (0x0B80, 0x0BFF, "ta"),  # Tamil       → Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu      → Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada     → Kannada
]

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "bn": "Bengali (বাংলা)",
}
INDIC_LANGS = {"hi", "ta", "te", "kn", "bn"}

# Characters that mark a natural TTS flush boundary
_SENTENCE_BOUNDARIES = frozenset(".?!।॥،؟\n")


def detect_language(text: str) -> str:
    """Return ISO lang code by scanning Unicode codepoints; defaults to 'en'."""
    for ch in text:
        cp = ord(ch)
        for start, end, lang in _INDIC_SCRIPT_RANGES:
            if start <= cp <= end:
                return lang
    return "en"


# -----------------------------------------------------------------------------
# 3. Sentence Chunker — enables TTS to start before full LLM response arrives
# -----------------------------------------------------------------------------
class SentenceChunker:
    """
    Accumulates LLM tokens and yields a flush whenever a sentence boundary
    appears or the token buffer reaches max_tokens. This cuts perceived latency:
    the first audio chunk plays while the model is still generating the rest.
    """
    def __init__(self, max_tokens: int = 15):
        self.max_tokens = max_tokens
        self._buf: List[str] = []

    def push(self, token: str) -> Optional[str]:
        self._buf.append(token)
        text = "".join(self._buf)
        if any(ch in text for ch in _SENTENCE_BOUNDARIES) or len(self._buf) >= self.max_tokens:
            self._buf = []
            return text.strip()
        return None

    def flush(self) -> Optional[str]:
        if self._buf:
            text = "".join(self._buf).strip()
            self._buf = []
            return text or None
        return None


# -----------------------------------------------------------------------------
# 4. Mock STT, TTS, and LLM Services (English baseline)
# -----------------------------------------------------------------------------
class MockSTT:
    _PROMPTS = [
        "explain consistency in database clusters",
        "tell me a short story about a fast computer",
        "what is the distance to the moon",
        "why is the sky blue",
    ]

    async def transcribe(self, audio_data_len: int) -> str:
        await asyncio.sleep(0.08)
        import random
        return random.choice(self._PROMPTS)


class MockLLM:
    def __init__(self, name: str, setup_delay: float, token_delay: float, text: str):
        self.name = name
        self.setup_delay = setup_delay
        self.token_delay = token_delay
        self.tokens = text.split(" ")

    async def stream_tokens(self, prompt: str):
        await asyncio.sleep(self.setup_delay)
        for i, token in enumerate(self.tokens):
            yield token if i == 0 else " " + token
            await asyncio.sleep(self.token_delay)


class MockTTS:
    async def stream_audio_chunks(self, token_queue: asyncio.Queue):
        while True:
            token = await token_queue.get()
            if token is None:
                token_queue.task_done()
                break
            for _ in range(3):
                await asyncio.sleep(0.01)
                yield b"\x55\xaa" * 512  # 1 024 bytes mock PCM
            token_queue.task_done()


_PRIMARY_TEXT = (
    "This is a detailed response from the Primary LLM. "
    "It contains thorough details and is optimized for comprehensive understanding."
)
_FALLBACK_TEXT = "Fast SLM backup: Sorry, Primary was slow. I am answering instead."

primary_llm = MockLLM("Primary-LLM-Remote", setup_delay=0.05, token_delay=0.02, text=_PRIMARY_TEXT)
fallback_slm = MockLLM("Fallback-SLM-Local", setup_delay=0.01, token_delay=0.005, text=_FALLBACK_TEXT)
stt_service = MockSTT()
tts_service = MockTTS()


# -----------------------------------------------------------------------------
# 5. Indic Mock Services — realistic latency profiles
# -----------------------------------------------------------------------------
class MockIndicSTT:
    """
    Indic STT is ~110 ms vs ~80 ms for English because Indic ASR models
    (IndicWhisper, Sarvam Saarika) have larger decoder vocabularies.
    """
    _PROMPTS: Dict[str, List[str]] = {
        "hi": [
            "डेटाबेस में consistency क्या होती है?",
            "मुझे cloud computing के बारे में बताओ",
            "machine learning कैसे काम करती है?",
        ],
        "ta": [
            "தரவுத்தளத்தில் consistency என்றால் என்ன?",
            "cloud computing பற்றி சொல்லுங்கள்",
        ],
        "te": [
            "డేటాబేస్‌లో consistency అంటే ఏమిటి?",
            "cloud computing గురించి చెప్పండి",
        ],
        "kn": [
            "ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ consistency ಎಂದರೇನು?",
            "cloud computing ಬಗ್ಗೆ ಹೇಳಿ",
        ],
        "bn": [
            "ডেটাবেসে consistency কী?",
            "cloud computing সম্পর্কে বলুন",
        ],
    }

    async def transcribe(self, audio_data_len: int, lang: str) -> str:
        await asyncio.sleep(0.11)
        import random
        return random.choice(self._PROMPTS.get(lang, self._PROMPTS["hi"]))


class MockIndicLLM:
    """
    Indic tokenization is 3-4× less efficient than English in most LLMs,
    so TTFT is ~250 ms and per-token delay is ~25 ms (more tokens per word).
    Responses mix native script with English technical terms — realistic code-switching.
    """
    _RESPONSES: Dict[str, str] = {
        "hi": (
            "डेटाबेस में consistency का अर्थ है कि सभी nodes एक ही data देखते हैं। "
            "CAP theorem के अनुसार, distributed system में consistency, availability, "
            "और partition tolerance तीनों एक साथ guarantee नहीं हो सकते। "
            "इसलिए engineers को अपने use-case के हिसाब से trade-off चुनना पड़ता है।"
        ),
        "ta": (
            "தரவுத்தளத்தில் consistency என்பது அனைத்து nodes-உம் ஒரே data-ஐ பார்க்கின்றன என்று பொருள். "
            "CAP theorem படி, distributed system-ல் consistency மற்றும் availability-ல் ஒன்றை "
            "தேர்வு செய்ய வேண்டும். இது system design-ல் மிக முக்கியமான trade-off ஆகும்."
        ),
        "te": (
            "డేటాబేస్‌లో consistency అంటే అన్ని nodes ఒకే data చూస్తాయి. "
            "CAP theorem ప్రకారం, distributed system లో consistency మరియు availability లో "
            "ఒకదాన్ని ఎంచుకోవాలి. ఇది system design లో చాలా కీలకమైన trade-off."
        ),
        "kn": (
            "ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ consistency ಎಂದರೆ ಎಲ್ಲಾ nodes ಒಂದೇ data ನೋಡುತ್ತವೆ. "
            "CAP theorem ಪ್ರಕಾರ, distributed system ನಲ್ಲಿ consistency ಮತ್ತು availability ನಲ್ಲಿ "
            "ಒಂದನ್ನು ಆರಿಸಬೇಕು. ಇದು system design ನಲ್ಲಿ ಬಹಳ ಮುಖ್ಯವಾದ trade-off."
        ),
        "bn": (
            "ডেটাবেসে consistency মানে হলো সব nodes একই data দেখে। "
            "CAP theorem অনুযায়ী, distributed system-এ consistency এবং availability-র মধ্যে "
            "একটি বেছে নিতে হবে। এটি system design-এর একটি অত্যন্ত গুরুত্বপূর্ণ trade-off।"
        ),
    }

    def __init__(self):
        self.name = "Indic-LLM"
        self.setup_delay = 0.25   # 250 ms TTFT — Indic tokenization overhead
        self.token_delay = 0.025  # 25 ms/token

    async def stream_tokens(self, prompt: str, lang: str = "hi"):
        await asyncio.sleep(self.setup_delay)
        text = self._RESPONSES.get(lang, self._RESPONSES["hi"])
        tokens = text.split(" ")
        for i, token in enumerate(tokens):
            yield token if i == 0 else " " + token
            await asyncio.sleep(self.token_delay)


class MockIndicTTS:
    """
    Indic TTS (e.g. Sarvam Bulbul, IndicTTS) runs ~40 ms/chunk vs ~30 ms for
    English due to longer phoneme sequences. Chunk size is slightly larger too.
    """
    async def stream_audio_chunks(self, text_chunk: str, lang: str):
        await asyncio.sleep(0.04)
        yield b"\xaa\x55" * 768  # 1 536 bytes mock PCM (Indic phoneme burst)


indic_stt = MockIndicSTT()
indic_llm = MockIndicLLM()
indic_tts = MockIndicTTS()

# -----------------------------------------------------------------------------
# 6. Adaptive TTFT Tracking & Smart Fallback Router
# -----------------------------------------------------------------------------
metrics = {
    "primary_hits": 0,
    "fallback_hits": 0,
    "lang_distribution": defaultdict(int),
}

# Rolling window of observed TTFT samples per language (capped at 50)
_ttft_samples: Dict[str, List[float]] = defaultdict(list)
_TTFT_SAMPLE_CAP = 50
_TTFT_BASE = {"en": 0.200, "indic": 0.350}  # Indic gets a wider base window


def get_adaptive_timeout(lang: str) -> float:
    """
    Returns a dynamic TTFT timeout derived from recent p90 observations.
    Falls back to a language-appropriate base if fewer than 5 samples exist.
    Indic base is 350 ms because tokenization overhead raises typical TTFT.
    """
    base = _TTFT_BASE["indic"] if lang in INDIC_LANGS else _TTFT_BASE["en"]
    samples = _ttft_samples[lang]
    if len(samples) >= 5:
        p90 = float(np.percentile(samples, 90))
        return max(p90 * 1.5, base)
    return base


def _record_ttft(lang: str, elapsed: float):
    samples = _ttft_samples[lang]
    samples.append(elapsed)
    if len(samples) > _TTFT_SAMPLE_CAP:
        samples.pop(0)


async def stream_llm_with_fallback_lang(prompt: str, lang: str):
    """Language-aware LLM routing with adaptive TTFT timeout."""
    timeout = get_adaptive_timeout(lang)
    start = time.time()

    if lang in INDIC_LANGS:
        primary_gen = indic_llm.stream_tokens(prompt, lang)
    else:
        primary_gen = primary_llm.stream_tokens(prompt)

    fallback_gen = fallback_slm.stream_tokens(prompt)

    try:
        first_token = await asyncio.wait_for(primary_gen.__anext__(), timeout=timeout)
        elapsed = time.time() - start
        _record_ttft(lang, elapsed)
        metrics["primary_hits"] += 1
        logger.info(f"[Route] [{lang.upper()}] Primary TTFT={elapsed:.3f}s (limit={timeout:.3f}s)")
        yield first_token
        async for token in primary_gen:
            yield token

    except (asyncio.TimeoutError, StopAsyncIteration):
        elapsed = time.time() - start
        metrics["fallback_hits"] += 1
        logger.warning(
            f"[Route] [{lang.upper()}] Primary slow ({elapsed:.3f}s > {timeout:.3f}s). "
            "Falling back to SLM."
        )
        async for token in fallback_gen:
            yield token


# Keep the original for backwards-compatibility with pressure/efficiency tests
async def stream_llm_with_fallback(prompt: str, timeout: float = 0.200):
    async for token in stream_llm_with_fallback_lang(prompt, "en"):
        yield token


# -----------------------------------------------------------------------------
# 7. Session State Management
# -----------------------------------------------------------------------------
class VoiceSession:
    def __init__(self, session_id: str, worker: str):
        self.session_id = session_id
        self.worker = worker
        self.state = "Listening"
        self.language = "en"
        self.history: List[Dict[str, str]] = []
        self.active_task: Optional[asyncio.Task] = None
        self.generated_tokens: List[str] = []
        self.sent_tokens_count = 0
        self.audio_bytes_received = 0
        self.connected = False
        self.created_at = time.time()
        self.turn_count = 0

    def reset_pipeline(self):
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        self.generated_tokens = []
        self.sent_tokens_count = 0
        self.audio_bytes_received = 0

    def handle_barge_in(self) -> Optional[str]:
        if self.state == "Listening":
            return None

        prev_state = self.state
        self.reset_pipeline()

        spoken_text = ""
        if self.generated_tokens:
            end_idx = min(self.sent_tokens_count, len(self.generated_tokens))
            spoken_text = "".join(self.generated_tokens[:end_idx]).strip()
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["content"] = spoken_text + " [INTERRUPTED]"
            elif spoken_text:
                self.history.append({"role": "assistant", "content": spoken_text + " [INTERRUPTED]"})

        self.state = "Listening"
        return f"Barge-in: interrupted {prev_state}. Truncated to: '{spoken_text}'"


# Sessions persist across reconnects so KV-cache affinity is preserved.
active_sessions: Dict[str, VoiceSession] = {}
_SESSION_TTL = 3600


def get_or_create_session(session_id: str) -> VoiceSession:
    now = time.time()
    expired = [
        sid for sid, s in active_sessions.items()
        if not s.connected and (now - s.created_at) > _SESSION_TTL
    ]
    for sid in expired:
        del active_sessions[sid]
        logger.info(f"Evicted expired session {sid}")

    if session_id not in active_sessions:
        worker = balancer.get_node(session_id)
        active_sessions[session_id] = VoiceSession(session_id, worker)
        logger.info(f"New session {session_id} → {worker}")
    else:
        s = active_sessions[session_id]
        logger.info(f"Resumed session {session_id} (turns={s.turn_count}, lang={s.language})")
    return active_sessions[session_id]


# -----------------------------------------------------------------------------
# 8. WebSocket Endpoint
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id", f"session-{int(time.time() * 1000)}")

    session = get_or_create_session(session_id)
    session.connected = True
    gpu_worker = session.worker
    logger.info(f"Connected: {session_id} → {gpu_worker} (turns={session.turn_count})")

    async def send_json_msg(event: str, data: str):
        try:
            await websocket.send_json({"event": event, "data": data})
        except Exception:
            pass

    await send_json_msg("session_init", json.dumps({
        "worker": gpu_worker,
        "session_id": session_id,
        "resumed": session.turn_count > 0,
        "history_turns": session.turn_count,
        "language": session.language,
    }))

    async def run_pipeline(prompt_override: Optional[str] = None, lang_hint: str = "auto"):
        try:
            session.state = "Thinking"
            await send_json_msg("status", "Transcribing Speech...")

            # --- STT ---
            prompt = prompt_override
            if not prompt:
                effective_lang = lang_hint if lang_hint in INDIC_LANGS else "en"
                if effective_lang in INDIC_LANGS:
                    prompt = await indic_stt.transcribe(session.audio_bytes_received, effective_lang)
                else:
                    prompt = await stt_service.transcribe(session.audio_bytes_received)

            # --- Language Detection ---
            if lang_hint in LANG_NAMES and lang_hint != "auto":
                lang = lang_hint
            else:
                lang = detect_language(prompt)

            session.language = lang
            metrics["lang_distribution"][lang] += 1

            logger.info(f"[Orchestrator] {session_id} lang={lang} prompt='{prompt}'")
            session.history.append({"role": "user", "content": prompt})
            await send_json_msg("transcription", prompt)
            await send_json_msg("language_detected", json.dumps({
                "lang": lang,
                "name": LANG_NAMES[lang],
                "adaptive_timeout_ms": int(get_adaptive_timeout(lang) * 1000),
            }))

            # --- LLM → Chunked TTS pipeline ---
            await send_json_msg("status", f"Querying {LANG_NAMES[lang]} LLM...")
            session.state = "Speaking"
            await send_json_msg("status", "Streaming Audio Out...")

            chunker = SentenceChunker(max_tokens=15)

            async def flush_to_tts(text: str):
                if lang in INDIC_LANGS:
                    async for audio in indic_tts.stream_audio_chunks(text, lang):
                        await websocket.send_bytes(audio)
                        session.sent_tokens_count = len(session.generated_tokens)
                else:
                    tq: asyncio.Queue = asyncio.Queue()
                    await tq.put(text)
                    await tq.put(None)
                    async for audio in tts_service.stream_audio_chunks(tq):
                        await websocket.send_bytes(audio)
                        session.sent_tokens_count = len(session.generated_tokens)

            async for token in stream_llm_with_fallback_lang(prompt, lang):
                session.generated_tokens.append(token)
                chunk = chunker.push(token)
                if chunk:
                    await flush_to_tts(chunk)

            remaining = chunker.flush()
            if remaining:
                await flush_to_tts(remaining)

            full_response = "".join(session.generated_tokens)
            session.history.append({"role": "assistant", "content": full_response})
            session.turn_count += 1
            session.state = "Listening"
            await send_json_msg("status", "Finished speaking.")
            await send_json_msg("assistant_response", full_response)
            session.reset_pipeline()

        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled: {session_id}")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            session.state = "Listening"
            await send_json_msg("status", f"Pipeline error: {str(e)}")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                if session.state != "Listening":
                    log_msg = session.handle_barge_in()
                    if log_msg:
                        logger.info(f"[Barge-in] {log_msg}")
                        await send_json_msg("barge_in_triggered", log_msg)
                session.audio_bytes_received += len(message["bytes"])

            elif "text" in message:
                data = json.loads(message["text"])
                event = data.get("event")

                if event in ["barge_in", "speech_start"]:
                    log_msg = session.handle_barge_in()
                    if log_msg:
                        logger.info(f"[Control Barge-in] {log_msg}")
                        await send_json_msg("barge_in_triggered", log_msg)

                elif event == "speech_end":
                    lang_hint = data.get("language", "auto")
                    session.reset_pipeline()
                    pipeline_task = asyncio.create_task(
                        run_pipeline(data.get("prompt"), lang_hint)
                    )
                    session.active_task = pipeline_task

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"Disconnected: {session_id} (history preserved, turns={session.turn_count})")
        session.reset_pipeline()
    finally:
        session.connected = False
        session.state = "Listening"


# -----------------------------------------------------------------------------
# 9. Admin Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_endpoint():
    connected = sum(1 for s in active_sessions.values() if s.connected)
    ttft_stats = {}
    for lang, samples in _ttft_samples.items():
        if samples:
            ttft_stats[lang] = {
                "p50_ms": round(float(np.percentile(samples, 50)) * 1000, 1),
                "p90_ms": round(float(np.percentile(samples, 90)) * 1000, 1),
                "adaptive_timeout_ms": round(get_adaptive_timeout(lang) * 1000, 1),
                "samples": len(samples),
            }
    return {
        "status": "healthy",
        "active_connections": connected,
        "total_sessions": len(active_sessions),
        "primary_llm_hits": metrics["primary_hits"],
        "fallback_slm_hits": metrics["fallback_hits"],
        "primary_llm_delay_ms": int(primary_llm.setup_delay * 1000),
        "fallback_slm_delay_ms": int(fallback_slm.setup_delay * 1000),
        "lang_distribution": dict(metrics["lang_distribution"]),
        "ttft_by_lang": ttft_stats,
    }


@app.get("/sessions")
def sessions_endpoint():
    worker_stats: Dict[str, Dict] = {
        node: {"sessions": 0, "connected": 0, "total_turns": 0}
        for node in workers
    }
    for s in active_sessions.values():
        if s.worker in worker_stats:
            worker_stats[s.worker]["sessions"] += 1
            worker_stats[s.worker]["total_turns"] += s.turn_count
            if s.connected:
                worker_stats[s.worker]["connected"] += 1
    return {
        "worker_routing": worker_stats,
        "session_list": [
            {
                "session_id": s.session_id,
                "worker": s.worker,
                "state": s.state,
                "language": s.language,
                "connected": s.connected,
                "turns": s.turn_count,
                "history_len": len(s.history),
            }
            for s in active_sessions.values()
        ],
    }


@app.get("/toggle-latency", response_class=PlainTextResponse)
def toggle_latency():
    old = primary_llm.setup_delay
    new = 0.05 if old == 0.40 else 0.40
    primary_llm.setup_delay = new
    msg = f"Primary LLM delay: {old}s → {new}s (threshold=0.200s English / 0.350s Indic)"
    logger.info(f"[Admin] {msg}")
    return msg


@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Voice Gateway: templates/index.html missing</h1>"
