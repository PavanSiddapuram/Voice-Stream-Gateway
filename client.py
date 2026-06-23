import asyncio
import json
import httpx
import websockets
import time

server_addr = "localhost:8000"
session_id = "py-demo-session"

async def toggle_latency():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{server_addr}/toggle-latency")
        print(f"   Server Admin: {resp.text}")

async def run_scenario(scenario_name: str, simulate_barge_in: bool):
    print(f"\n[SCENARIO] Starting Scenario: {scenario_name}")
    url = f"ws://{server_addr}/ws?session_id={session_id}"
    
    async with websockets.connect(url) as ws:
        print("   [CONNECTED] Connected to Gateway WebSocket")
        
        audio_frame_count = 0
        barge_in_done = False
        start_time = time.time()
        
        async def read_responses():
            nonlocal audio_frame_count, barge_in_done
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        audio_frame_count += 1
                        
                        # Simulate Interruption (Barge-in)
                        if simulate_barge_in and audio_frame_count == 5 and not barge_in_done:
                            barge_in_done = True
                            print("   [BARGE-IN] User interrupts! Sending 'speech_start' (Barge-in command)...")
                            # Send control barge-in message
                            await ws.send(json.dumps({"event": "speech_start"}))
                            # Send user audio chunks
                            for _ in range(3):
                                await ws.send(b"\x00" * 1024)
                                await asyncio.sleep(0.03)
                    else:
                        resp = json.loads(message)
                        event = resp.get("event")
                        data = resp.get("data")
                        
                        if event == "status":
                            print(f"   [STATUS UPDATE]: {data}")
                        elif event == "transcription":
                            print(f"   [STT OUTPUT]: '{data}'")
                        elif event == "assistant_response":
                            print(f"   [FULL RESPONSE]: '{data}'")
                        elif event == "barge_in_triggered":
                            print(f"   [BARGE-IN CONFIRMED]: {data}")
                            
            except websockets.exceptions.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(read_responses())
        
        # 1. Simulate speech stream
        print("   [MIC] Speaking... (Streaming audio chunks)")
        for _ in range(5):
            await ws.send(b"\x00" * 1024)
            await asyncio.sleep(0.05)
            
        # 2. Finished speaking
        print("   [MIC] Speech finished. Sending speech_end command...")
        await ws.send(json.dumps({
            "event": "speech_end",
            "prompt": "explain consistency in database clusters"
        }))
        
        # Wait for processing to complete (or for client reader task to finish)
        await asyncio.sleep(2.5)
        await ws.close()
        await reader_task
        
        elapsed = time.time() - start_time
        print(f"   [FINISH] Scenario finished in {elapsed:.2f}s. Audio chunks received: {audio_frame_count}")

async def main():
    print("=================================================================")
    print("  Running Automated Python Voice Gateway Simulation Client")
    print(f"   Target Gateway: {server_addr}")
    print(f"   Session ID:     {session_id}")
    print("=================================================================\n")

    # Phase 1: Test Fast Route (Primary LLM returns fast)
    await run_scenario("Fast Primary LLM Scenario", simulate_barge_in=False)
    
    await asyncio.sleep(1)

    # Phase 2: Test Slow Route -> Fallback SLM
    print("\n[LATENCY] Degrading primary LLM network connection (setting delay to 400ms)...")
    await toggle_latency()
    await run_scenario("Fallback SLM Scenario", simulate_barge_in=False)

    # Restore primary latency
    print("\n[LATENCY] Restoring primary LLM network connection (setting delay back to 50ms)...")
    await toggle_latency()

    await asyncio.sleep(1)

    # Phase 3: Test Barge-in (Interruption) handling
    await run_scenario("Barge-in Interruption Scenario", simulate_barge_in=True)

    print("\n[SUCCESS] All scenarios complete.")

if __name__ == "__main__":
    asyncio.run(main())
