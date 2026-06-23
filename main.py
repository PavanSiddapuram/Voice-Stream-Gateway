import asyncio
import json
import time
import hashlib
import bisect
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
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
# 2. Mock STT, TTS, and LLM Services
# -----------------------------------------------------------------------------
class MockSTT:
    async def transcribe(self, audio_data_len: int) -> str:
        # Simulate STT processing delay
        await asyncio.sleep(0.08)
        prompts = [
            "explain consistency in database clusters",
            "tell me a short story about a fast computer",
            "what is the distance to the moon",
            "why is the sky blue"
        ]
        import random
        return random.choice(prompts)

class MockLLM:
    def __init__(self, name: str, setup_delay: float, token_delay: float, text: str):
        self.name = name
        self.setup_delay = setup_delay
        self.token_delay = token_delay
        self.tokens = text.split(" ")

    async def stream_tokens(self, prompt: str):
        # Simulate TTFT (Time-To-First-Token) latency
        await asyncio.sleep(self.setup_delay)
        for i, token in enumerate(self.tokens):
            yield token if i == 0 else " " + token
            await asyncio.sleep(self.token_delay)

class MockTTS:
    async def stream_audio_chunks(self, token_queue: asyncio.Queue):
        # For each token, simulate generating and streaming audio chunks
        while True:
            token = await token_queue.get()
            if token is None:
                token_queue.task_done()
                break
            
            # Generate 3 mock audio chunks per token with delay
            for _ in range(3):
                await asyncio.sleep(0.01) # synthesis delay
                # 1024 bytes representing mock PCM audio
                dummy_pcm = b"\x55\xaa" * 512
                yield dummy_pcm
            token_queue.task_done()

primary_text = "This is a detailed response from the Primary LLM. It contains thorough details and is optimized for comprehensive understanding."
fallback_text = "Fast SLM backup: Sorry, Primary was slow. I am answering instead."

primary_llm = MockLLM("Primary-LLM-Remote", setup_delay=0.05, token_delay=0.02, text=primary_text)
fallback_slm = MockLLM("Fallback-SLM-Local", setup_delay=0.01, token_delay=0.005, text=fallback_text)
stt_service = MockSTT()
tts_service = MockTTS()

# -----------------------------------------------------------------------------
# 3. Smart Fallback Router
# -----------------------------------------------------------------------------
metrics = {
    "primary_hits": 0,
    "fallback_hits": 0
}

async def stream_llm_with_fallback(prompt: str, timeout: float = 0.200):
    start_time = time.time()
    primary_gen = primary_llm.stream_tokens(prompt)
    fallback_gen = fallback_slm.stream_tokens(prompt)

    # We try to get the first token from Primary within the timeout window
    try:
        # Get first token from primary
        first_token = await asyncio.wait_for(primary_gen.__anext__(), timeout=timeout)
        elapsed = time.time() - start_time
        metrics["primary_hits"] += 1
        logger.info(f"[Route Decision] Routed to Primary LLM (First token in {elapsed:.3f}s)")
        yield first_token
        
        async for token in primary_gen:
            yield token

    except (asyncio.TimeoutError, StopAsyncIteration):
        # Primary timed out or errored out before yielding first token
        elapsed = time.time() - start_time
        metrics["fallback_hits"] += 1
        logger.warning(f"[Route Decision] Primary slow/failed ({elapsed:.3f}s). Routing to Fallback SLM.")
        
        async for token in fallback_gen:
            yield token

# -----------------------------------------------------------------------------
# 4. Session State Management
# -----------------------------------------------------------------------------
class VoiceSession:
    def __init__(self, session_id: str, worker: str):
        self.session_id = session_id
        self.worker = worker
        self.state = "Listening" # Listening, Thinking, Speaking
        self.history: List[Dict[str, str]] = []
        self.active_task: Optional[asyncio.Task] = None
        self.generated_tokens: List[str] = []
        self.sent_tokens_count = 0
        self.audio_bytes_received = 0

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
        
        # Truncate assistant response history
        spoken_text = ""
        if self.generated_tokens:
            end_idx = min(self.sent_tokens_count, len(self.generated_tokens))
            spoken_text = "".join(self.generated_tokens[:end_idx]).strip()
            
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["content"] = spoken_text + " [INTERRUPTED]"
            elif spoken_text:
                self.history.append({"role": "assistant", "content": spoken_text + " [INTERRUPTED]"})

        self.state = "Listening"
        return f"Barge-in: interrupted {prev_state}. Truncated response to: '{spoken_text}'"

active_sessions: Dict[str, VoiceSession] = {}

# -----------------------------------------------------------------------------
# 5. WebSocket Connection & Server
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Reset primary LLM delay to fast (50ms) on new connections
    primary_llm.setup_delay = 0.05
    session_id = websocket.query_params.get("session_id", f"session-{int(time.time() * 1000)}")
    
    # Route session sticky to worker
    gpu_worker = balancer.get_node(session_id)
    if session_id not in active_sessions:
        active_sessions[session_id] = VoiceSession(session_id, gpu_worker)
    
    session = active_sessions[session_id]
    logger.info(f"Client connected. SessionID: {session_id} routed to warm worker: {gpu_worker}")

    async def send_json_msg(event: str, data: str):
        try:
            await websocket.send_json({"event": event, "data": data})
        except Exception:
            pass

    async def run_pipeline(prompt_override: Optional[str] = None):
        try:
            # 1. STT Phase
            session.state = "Thinking"
            await send_json_msg("status", "Transcribing Speech...")
            prompt = prompt_override
            if not prompt:
                prompt = await stt_service.transcribe(session.audio_bytes_received)
            
            logger.info(f"[Orchestrator] Session {session_id} prompt: '{prompt}'")
            session.history.append({"role": "user", "content": prompt})
            await send_json_msg("transcription", prompt)

            # 2. LLM Route & TTS Queue
            await send_json_msg("status", "Querying LLM...")
            token_queue = asyncio.Queue()
            
            async def feed_tokens():
                try:
                    async for token in stream_llm_with_fallback(prompt):
                        session.generated_tokens.append(token)
                        await token_queue.put(token)
                finally:
                    await token_queue.put(None)

            feeder_task = asyncio.create_task(feed_tokens())

            # 3. TTS & Streaming Out
            session.state = "Speaking"
            await send_json_msg("status", "Streaming Audio Out...")
            
            audio_generator = tts_service.stream_audio_chunks(token_queue)
            
            try:
                async for audio_chunk in audio_generator:
                    await websocket.send_bytes(audio_chunk)
                    session.sent_tokens_count = len(session.generated_tokens)
            except Exception as e:
                logger.error(f"Error during audio stream: {e}")
                raise e

            await feeder_task
            
            # Successful completion
            full_response = "".join(session.generated_tokens)
            session.history.append({"role": "assistant", "content": full_response})
            session.state = "Listening"
            await send_json_msg("status", "Finished speaking.")
            await send_json_msg("assistant_response", full_response)
            session.reset_pipeline()

        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            session.state = "Listening"
            await send_json_msg("status", f"Pipeline error: {str(e)}")

    try:
        while True:
            # Wait for user input (audio chunk or control frame)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary speech frame
                if session.state != "Listening":
                    log_msg = session.handle_barge_in()
                    if log_msg:
                        logger.info(f"[Barge-in] {log_msg}")
                        await send_json_msg("barge_in_triggered", log_msg)
                
                session.audio_bytes_received += len(message["bytes"])
                
            elif "text" in message:
                data = json.loads(message["text"])
                event = data.get("event")
                prompt_override = data.get("prompt")

                if event in ["barge_in", "speech_start"]:
                    log_msg = session.handle_barge_in()
                    if log_msg:
                        logger.info(f"[Control Barge-in] {log_msg}")
                        await send_json_msg("barge_in_triggered", log_msg)

                elif event == "speech_end":
                    # Start async orchestration pipeline
                    session.reset_pipeline()
                    pipeline_task = asyncio.create_task(run_pipeline(prompt_override))
                    session.active_task = pipeline_task

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"Session {session_id} disconnected.")
        session.reset_pipeline()
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]

# -----------------------------------------------------------------------------
# 6. Admin Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_endpoint():
    return {
        "status": "healthy",
        "active_connections": len(active_sessions),
        "primary_llm_hits": metrics["primary_hits"],
        "fallback_slm_hits": metrics["fallback_hits"],
        "primary_llm_delay_ms": int(primary_llm.setup_delay * 1000),
        "fallback_slm_delay_ms": int(fallback_slm.setup_delay * 1000)
    }

@app.get("/toggle-latency", response_class=PlainTextResponse)
def toggle_latency():
    old_delay = primary_llm.setup_delay
    new_delay = 0.05 if old_delay == 0.40 else 0.40
    primary_llm.setup_delay = new_delay
    msg = f"Primary LLM setup delay updated from {old_delay}s to {new_delay}s (Threshold is 0.200s)"
    logger.info(f"[Admin] {msg}")
    return msg

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Voice Gateway Template index.html Missing</h1>"
