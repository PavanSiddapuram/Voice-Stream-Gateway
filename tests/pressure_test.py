import asyncio
import json
import time
import argparse
import websockets
import httpx
import numpy as np

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Voice Gateway Concurrency & Pressure Benchmark")
parser.add_argument("--concurrency", type=int, default=50, help="Number of concurrent sessions to spawn")
parser.add_argument("--barge-in-rate", type=float, default=0.2, help="Fraction of sessions that trigger barge-in (0.0 - 1.0)")
parser.add_argument("--server", type=str, default="127.0.0.1:8000", help="Voice Gateway address (host:port)")
args = parser.parse_args()

server_addr = args.server
concurrency = args.concurrency
barge_in_rate = args.barge_in_rate

# Metrics lists to collect stats
conn_latencies = []
ttft_latencies = []
barge_in_latencies = []
success_count = 0
error_count = 0
barge_in_success = 0

async def run_client_session(client_id: int):
    global success_count, error_count, barge_in_success
    session_id = f"pressure-test-client-{client_id}"
    url = f"ws://{server_addr}/ws?session_id={session_id}"
    
    # Decide if this client will barge in
    will_barge_in = np.random.rand() < barge_in_rate
    
    try:
        # 1. Connect and measure handshake latency
        t_start_conn = time.time()
        async with websockets.connect(url) as ws:
            conn_latencies.append(time.time() - t_start_conn)
            
            # 2. Stream user audio chunks (simulate talking)
            for _ in range(5):
                await ws.send(b"\x00" * 1024)
                await asyncio.sleep(0.04) # 40ms audio chunks
                
            # 3. Finished speaking, send trigger
            t_sent_trigger = time.time()
            await ws.send(json.dumps({
                "event": "speech_end",
                "prompt": "stress test benchmark prompt text payload"
            }))
            
            audio_frame_count = 0
            first_frame_received = False
            barge_in_sent_time = None
            
            # 4. Read response loop
            async for message in ws:
                if isinstance(message, bytes):
                    # Audio chunk received
                    if not first_frame_received:
                        first_frame_received = True
                        ttft_latencies.append(time.time() - t_sent_trigger)
                        
                    audio_frame_count += 1
                    
                    # If this client is a barge-in candidate, trigger barge-in after 3 audio chunks
                    if will_barge_in and audio_frame_count == 3 and barge_in_sent_time is None:
                        barge_in_sent_time = time.time()
                        # Send barge-in control message
                        await ws.send(json.dumps({"event": "speech_start"}))
                else:
                    resp = json.loads(message)
                    event = resp.get("event")
                    
                    if event == "barge_in_triggered":
                        if barge_in_sent_time:
                            barge_in_latencies.append(time.time() - barge_in_sent_time)
                            barge_in_success += 1
                        break
                    elif event == "assistant_response":
                        # Finished speaking response fully
                        break
            
            success_count += 1
            
    except Exception as e:
        error_count += 1

def print_stats_table(name: str, values: list):
    if not values:
        print(f"{name:<25} | No data collected")
        return
        
    vals = np.array(values) * 1000 # Convert to milliseconds
    p50 = np.percentile(vals, 50)
    p90 = np.percentile(vals, 90)
    p99 = np.percentile(vals, 99)
    mean_val = np.mean(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)
    
    print(f"{name:<25} | p50: {p50:6.1f}ms | p90: {p90:6.1f}ms | p99: {p99:6.1f}ms | Mean: {mean_val:6.1f}ms | Min: {min_val:5.1f}ms | Max: {max_val:6.1f}ms")

async def main():
    print("=================================================================")
    print("      Voice Gateway Concurrency & Pressure Benchmark             ")
    print(f"      Concurrently active connections: {concurrency}")
    print(f"      Target server:                   {server_addr}")
    print(f"      Barge-in probability:            {barge_in_rate * 100:.1f}%")
    print("=================================================================\n")

    t_start = time.time()
    
    # Create concurrent client tasks
    tasks = [run_client_session(i) for i in range(concurrency)]
    print(f"[BENCHMARK] Spawning {concurrency} concurrent client loops...")
    await asyncio.gather(*tasks)
    
    total_duration = time.time() - t_start
    throughput = concurrency / total_duration
    
    print("\n=================================================================")
    print("                        BENCHMARK SUMMARY                        ")
    print("=================================================================")
    print(f"Total Test Duration:       {total_duration:.2f} seconds")
    print(f"Successful Connections:    {success_count} / {concurrency}")
    print(f"Errored Connections:       {error_count} / {concurrency}")
    print(f"Barge-ins Triggered:       {barge_in_success}")
    print(f"Throughput:                {throughput:.2f} sessions/sec")
    print("-----------------------------------------------------------------")
    
    print_stats_table("WS Handshake Latency", conn_latencies)
    print_stats_table("Time-To-First-Token (TTFT)", ttft_latencies)
    print_stats_table("Barge-In Interruption Lag", barge_in_latencies)
    print("=================================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
