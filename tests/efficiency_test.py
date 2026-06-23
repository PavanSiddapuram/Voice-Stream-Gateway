import asyncio
import json
import httpx
import websockets
import argparse
import sys

parser = argparse.ArgumentParser(description="Voice Gateway Stability & Resource Leak Profiler")
parser.add_argument("--cycles", type=int, default=50, help="Number of connection cycles to run")
parser.add_argument("--server", type=str, default="127.0.0.1:8000", help="Voice Gateway address")
args = parser.parse_args()

server_addr = args.server
cycles = args.cycles

async def run_single_cycle(index: int):
    url = f"ws://{server_addr}/ws?session_id=leak-test-session-{index}"
    try:
        async with websockets.connect(url) as ws:
            # Send small audio frame
            await ws.send(b"\x00" * 1024)
            await asyncio.sleep(0.01)
            
            # Trigger STT and pipeline
            await ws.send(json.dumps({
                "event": "speech_end",
                "prompt": "stability test query"
            }))
            
            # Read until full text response
            async for message in ws:
                if not isinstance(message, bytes):
                    resp = json.loads(message)
                    if resp.get("event") == "assistant_response":
                        break
    except Exception:
        pass

async def query_health():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"http://{server_addr}/health")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Error querying /health endpoint: {e}")
    return None

async def main():
    print("=================================================================")
    print("      Voice Gateway Stability & Resource Leak Profiler           ")
    print(f"      Target server:                   {server_addr}")
    print(f"      Total connection cycles to run:  {cycles}")
    print("=================================================================\n")

    # 1. Query initial health
    initial_health = await query_health()
    if not initial_health:
        print("[ERROR] Gateway server is not running or unreachable on port 8000.")
        sys.exit(1)
        
    print(f"[START] Initial active connections on server: {initial_health['active_connections']}")
    print(f"[START] Primary hits: {initial_health['primary_llm_hits']}, Fallback hits: {initial_health['fallback_slm_hits']}")
    print("-" * 65)

    # 2. Run cycles
    print(f"[STABILITY] Launching {cycles} sequential connection lifecycle cycles...")
    for i in range(1, cycles + 1):
        await run_single_cycle(i)
        
        # Check health status every 10 cycles
        if i % 10 == 0:
            health = await query_health()
            if health:
                print(f"   Cycle {i:3d}/{cycles:3d} | Active WS Connections on Server: {health['active_connections']} (Expected: 0 or 1)")
            else:
                print(f"   Cycle {i:3d}/{cycles:3d} | Server connection failed")
            await asyncio.sleep(0.05)

    # 3. Final health check to inspect leakage
    print("-" * 65)
    print("[STABILITY] Cooling down server task queues...")
    await asyncio.sleep(1.0) # Wait for final tasks to finalize
    
    final_health = await query_health()
    if not final_health:
        print("[ERROR] Failed to query final server health state.")
        sys.exit(1)
        
    active_leaks = final_health["active_connections"]
    print(f"[FINAL] Final active connections on server: {active_leaks}")
    print(f"[FINAL] Total Primary LLM Hits:             {final_health['primary_llm_hits']}")
    print(f"[FINAL] Total Fallback SLM Hits:            {final_health['fallback_slm_hits']}")
    
    if active_leaks == 0:
        print("\n[SUCCESS] No resource leaks detected! Server successfully cleaned up all WS connection states.")
    else:
        print(f"\n[WARNING] Possible connection leak! {active_leaks} active connections remained in server memory.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
