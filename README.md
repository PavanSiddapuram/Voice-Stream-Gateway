# High-Concurrency Streaming Voice Gateway

A high-performance, real-time voice agent gateway orchestrating bidirectional audio streaming, intelligent routing, and barge-in mitigation. Built with Python (FastAPI/WebSockets) to demonstrate core architectural primitives for production-grade AI voice systems.

## Key Features

1. **Consistent Hash Balancer (Sticky Session Affinity)**
   - Routes sessions dynamically based on `session_id` using a consistent hash ring with virtual nodes.
   - Ensures users hit the same worker node across multiple turns, maximizing **Prefix Cache Hits** in the LLM/SLM inference engines.

2. **Smart SLA Routing & SLM Fallback**
   - Implements a 200ms primary LLM timeout threshold.
   - If the primary LLM is congested or network latency degrades, the gateway routes queries dynamically to a warm, local Small Language Model (SLM), guaranteeing sub-second response times.

3. **Intelligent Barge-In Context Truncation**
   - Monitors user interrupts. Upon interruption, active text generation and audio synthesis tasks are immediately cancelled to reclaim CPU/GPU resources.
   - Truncates history back to the exact point of interruption to prevent the agent from talking over the user.

4. **Connection Pool & Leak-Free Session Lifecycle**
   - Uses asynchronous queues and robust `try-finally` blocks to guarantee immediate memory and resource cleanup upon disconnects.

5. **Stunning Web Dashboard**
   - Pulse-ring animations synced to agent states (Listening, Thinking, Speaking).
   - Real-time conversation logs, latency toggles, and simulation tools (no-microphone test capabilities).

## Project Structure

```
├── main.py                 # FastAPI Gateway & WebSocket server
├── client.py               # Automated simulation client
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Portal UI Dashboard
├── tests/
│   ├── pressure_test.py    # Concurrency and latency (p50/p90/p99) benchmark suite
│   └── efficiency_test.py  # Leak-free memory and cleanup validator
└── .gitignore              # Git ignore configuration
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

3. **Open the Portal**:
   Navigate to `http://localhost:8000` in your web browser. Click **Start Conversation** and run the simulations!

## Running Benchmarks

### Concurrency and Latency Pressure Test
Spawn 50 concurrent WebSockets sessions and measure connection and TTFT metrics:
```bash
python tests/pressure_test.py --concurrency 50 --barge-in-rate 0.2
```

### Connection Cleanup and Stability Profiler
Verify resource cleanup and trace potential leaks:
```bash
python tests/efficiency_test.py
```
