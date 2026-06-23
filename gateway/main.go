// Voice Stream Gateway — Go edition
//
// This binary is the high-concurrency orchestration layer that sits in front
// of Sarvam AI's Saarika (STT), LLM, and Bulbul (TTS) inference services.
// It handles everything that must be fast and concurrent:
//
//   - WebSocket management         (goroutine-per-connection, no shared event loop)
//   - KV-cache-aware session routing (consistent hash ring → same GPU worker)
//   - Barge-in state machine        (context.Cancel, instant, contention-free)
//   - Adaptive TTFT fallback        (per-language rolling p90 threshold)
//   - PCM resampling + VAD          (tight loops, GIL-free)
//   - Session persistence across reconnects
//
// Inference is delegated to Python/CUDA services (vLLM, Saarika, Bulbul)
// via HTTP/gRPC. Set SARVAM_API_KEY to use real Sarvam endpoints.
//
// Usage:
//
//	go run . --port 8080
//	go run . --port 8080 --workers "node0:8001,node1:8002"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"voice-gateway/ring"
	"voice-gateway/router"
	"voice-gateway/sarvam"
	"voice-gateway/session"
	"voice-gateway/ws"
)

func main() {
	port := flag.Int("port", 8080, "Gateway listen port")
	workerList := flag.String("workers",
		"GPU-Worker-Node-0 (A100-80GB),GPU-Worker-Node-1 (A100-80GB),GPU-Worker-Node-2 (A100-80GB)",
		"Comma-separated GPU worker node names for consistent hash ring")
	flag.Parse()

	workers := strings.Split(*workerList, ",")
	for i := range workers {
		workers[i] = strings.TrimSpace(workers[i])
	}

	hashRing := ring.New(workers, 50)
	store := session.NewStore()
	tracker := router.NewTTFTTracker()
	metrics := router.NewMetrics()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	store.StartEviction(ctx)

	handler := &ws.Handler{
		Ring:    hashRing,
		Store:   store,
		Tracker: tracker,
		Metrics: metrics,
		Saarika: sarvam.NewSaarikaClient(),
		Bulbul:  sarvam.NewBulbulClient(),
	}

	mux := http.NewServeMux()
	mux.Handle("/ws", handler)
	mux.HandleFunc("/health", healthHandler(store, metrics, tracker, handler))
	mux.HandleFunc("/sessions", sessionsHandler(store, workers))
	mux.HandleFunc("/toggle-latency", toggleLatencyHandler(handler))

	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", *port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		<-ctx.Done()
		log.Println("Shutting down gateway...")
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(shutCtx)
	}()

	log.Printf("Go Voice Gateway listening on :%d", *port)
	log.Printf("Workers: %s", strings.Join(workers, ", "))
	log.Printf("Set SARVAM_API_KEY for real Saarika/Bulbul endpoints.")

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func healthHandler(
	store *session.Store,
	m *router.Metrics,
	t *router.TTFTTracker,
	h *ws.Handler,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		primary, fallback, dist := m.Snapshot()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":             "healthy",
			"runtime":            "go",
			"active_connections": store.ActiveCount(),
			"total_sessions":     store.TotalCount(),
			"primary_llm_hits":   primary,
			"fallback_slm_hits":  fallback,
			"lang_distribution":  dist,
			"ttft_by_lang":       t.Stats(),
			"slow_primary_mode":  h.SlowPrimary,
		})
	}
}

func sessionsHandler(store *session.Store, workers []string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		workerStats := make(map[string]map[string]int)
		for _, wk := range workers {
			workerStats[wk] = map[string]int{"sessions": 0, "connected": 0, "total_turns": 0}
		}
		var list []map[string]any
		for _, s := range store.Snapshot() {
			s.Lock()
			if ws, ok := workerStats[s.Worker]; ok {
				ws["sessions"]++
				ws["total_turns"] += s.TurnCount
				if s.Connected {
					ws["connected"]++
				}
			}
			list = append(list, map[string]any{
				"session_id":  s.ID,
				"worker":      s.Worker,
				"state":       s.State,
				"language":    s.Language,
				"connected":   s.Connected,
				"turns":       s.TurnCount,
				"history_len": len(s.History),
			})
			s.Unlock()
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"worker_routing": workerStats,
			"session_list":   list,
		})
	}
}

func toggleLatencyHandler(h *ws.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h.SlowPrimary = !h.SlowPrimary
		state := "FAST (50ms TTFT)"
		if h.SlowPrimary {
			state = "SLOW (400ms TTFT — triggers SLM fallback)"
		}
		msg := "Primary LLM mode: " + state
		log.Printf("[admin] %s", msg)
		fmt.Fprintln(w, msg)
	}
}

func init() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)
	log.SetOutput(os.Stdout)
}
