// Benchmark: Go gateway vs Python gateway at increasing concurrency.
//
// Spawns N concurrent WebSocket sessions, measures WS handshake latency,
// Time-To-First-Token (audio), and barge-in response time.
// Outputs a side-by-side table when run against both gateways.
//
// Usage:
//
//	go run bench.go --target ws://localhost:8080 --concurrency 10000
//	go run bench.go --target ws://localhost:8000 --concurrency 500   # Python
//	go run bench.go --compare  # runs both and prints comparison table
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

type result struct {
	connLatency  float64 // seconds
	ttft         float64 // seconds, time-to-first-audio-byte
	bargeInLag   float64 // seconds, time from barge-in send to ack
	success      bool
	bargeInDone  bool
}

func runSession(id int, target string, bargeIn bool) result {
	r := result{}
	sessionID := fmt.Sprintf("bench-session-%d-%d", time.Now().UnixNano(), id)
	u, _ := url.Parse(target)
	q := u.Query()
	q.Set("session_id", sessionID)
	u.RawQuery = q.Encode()

	t0 := time.Now()
	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		return r
	}
	defer conn.Close()
	r.connLatency = time.Since(t0).Seconds()

	// Stream 5 fake PCM audio frames (simulates talking).
	for i := 0; i < 5; i++ {
		conn.WriteMessage(websocket.BinaryMessage, make([]byte, 1024))
		time.Sleep(40 * time.Millisecond)
	}

	// Trigger the pipeline.
	tTrigger := time.Now()
	conn.WriteJSON(map[string]string{
		"event":    "speech_end",
		"prompt":   "benchmark: explain KV-cache in transformer architectures",
		"language": "en",
	})

	audioFrames := 0
	firstAudio := false
	var bargeInSent time.Time

	conn.SetReadDeadline(time.Now().Add(10 * time.Second))

	for {
		msgType, payload, err := conn.ReadMessage()
		if err != nil {
			break
		}

		if msgType == websocket.BinaryMessage {
			if !firstAudio {
				firstAudio = true
				r.ttft = time.Since(tTrigger).Seconds()
			}
			audioFrames++
			// Trigger barge-in after 3rd audio chunk.
			if bargeIn && audioFrames == 3 && bargeInSent.IsZero() {
				bargeInSent = time.Now()
				conn.WriteJSON(map[string]string{"event": "speech_start"})
			}
		} else {
			var msg map[string]string
			json.Unmarshal(payload, &msg)
			switch msg["event"] {
			case "barge_in_triggered":
				if !bargeInSent.IsZero() {
					r.bargeInLag = time.Since(bargeInSent).Seconds()
					r.bargeInDone = true
				}
				r.success = true
				return r
			case "assistant_response":
				r.success = true
				return r
			}
		}
	}
	r.success = firstAudio
	return r
}

func runBatch(target string, concurrency int, bargeInRate float64) batchResult {
	results := make([]result, concurrency)
	var wg sync.WaitGroup
	var errCount int64

	sem := make(chan struct{}, concurrency)
	t0 := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			barge := float64(idx)/float64(concurrency) < bargeInRate
			results[idx] = runSession(idx, target, barge)
			if !results[idx].success {
				atomic.AddInt64(&errCount, 1)
			}
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(t0)

	var connLats, ttfts, bargeLags []float64
	successes, bargeIns := 0, 0
	for _, r := range results {
		if r.success {
			successes++
			if r.connLatency > 0 {
				connLats = append(connLats, r.connLatency)
			}
			if r.ttft > 0 {
				ttfts = append(ttfts, r.ttft)
			}
			if r.bargeInDone {
				bargeIns++
				bargeLags = append(bargeLags, r.bargeInLag)
			}
		}
	}

	return batchResult{
		concurrency:     concurrency,
		successes:       successes,
		errors:          int(errCount),
		bargeIns:        bargeIns,
		elapsed:         elapsed,
		throughput:      float64(concurrency) / elapsed.Seconds(),
		connStats:       computeStats(connLats),
		ttftStats:       computeStats(ttfts),
		bargeInStats:    computeStats(bargeLags),
	}
}

type stats struct {
	p50, p90, p99, mean float64
	min, max            float64
}

func computeStats(vals []float64) stats {
	if len(vals) == 0 {
		return stats{}
	}
	s := append([]float64{}, vals...)
	sort.Float64s(s)
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return stats{
		p50:  percentile(s, 50) * 1000,
		p90:  percentile(s, 90) * 1000,
		p99:  percentile(s, 99) * 1000,
		mean: sum / float64(len(s)) * 1000,
		min:  s[0] * 1000,
		max:  s[len(s)-1] * 1000,
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := p / 100.0 * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[lo]
	}
	return sorted[lo] + (idx-float64(lo))*(sorted[hi]-sorted[lo])
}

type batchResult struct {
	concurrency  int
	successes    int
	errors       int
	bargeIns     int
	elapsed      time.Duration
	throughput   float64
	connStats    stats
	ttftStats    stats
	bargeInStats stats
}

func printResult(label string, br batchResult) {
	fmt.Printf("\n%s\n", label)
	fmt.Printf("  Concurrency:        %d sessions\n", br.concurrency)
	fmt.Printf("  Successful:         %d / %d  (errors: %d)\n", br.successes, br.concurrency, br.errors)
	fmt.Printf("  Throughput:         %.1f sessions/sec\n", br.throughput)
	fmt.Printf("  Total duration:     %.2fs\n", br.elapsed.Seconds())
	fmt.Println()
	printStatsRow("WS Handshake", br.connStats)
	printStatsRow("TTFT (first audio)", br.ttftStats)
	if br.bargeIns > 0 {
		fmt.Printf("  Barge-ins success:  %d\n", br.bargeIns)
		printStatsRow("Barge-In Lag", br.bargeInStats)
	}
}

func printStatsRow(name string, s stats) {
	if s.p50 == 0 {
		fmt.Printf("  %-22s No data\n", name)
		return
	}
	fmt.Printf("  %-22s p50=%6.1fms  p90=%6.1fms  p99=%6.1fms  mean=%6.1fms\n",
		name, s.p50, s.p90, s.p99, s.mean)
}

func main() {
	target := flag.String("target", "ws://localhost:8080/ws", "WebSocket target URL")
	concurrency := flag.Int("concurrency", 100, "Number of concurrent sessions")
	bargeInRate := flag.Float64("barge-in-rate", 0.2, "Fraction of sessions that trigger barge-in")
	compare := flag.Bool("compare", false, "Compare Go (8080) vs Python (8000) gateways")
	flag.Parse()

	banner()

	if *compare {
		fmt.Println("Running comparison benchmark: Go Gateway vs Python Gateway")
		fmt.Println("(Ensure both servers are running before starting)")
		fmt.Println()

		levels := []int{100, 500, 1000, 2000, 5000}
		fmt.Printf("%-10s | %-20s %-20s | %-20s %-20s\n",
			"Sessions", "Go p50 TTFT", "Go p99 TTFT", "Python p50 TTFT", "Python p99 TTFT")
		fmt.Println(repeatStr("-", 85))

		for _, n := range levels {
			goRes := runBatch("ws://localhost:8080/ws", n, 0)
			pyRes := runBatch("ws://localhost:8000/ws", n, 0)
			fmt.Printf("%-10d | %-20s %-20s | %-20s %-20s\n",
				n,
				fmtMs(goRes.ttftStats.p50), fmtMs(goRes.ttftStats.p99),
				fmtMs(pyRes.ttftStats.p50), fmtMs(pyRes.ttftStats.p99),
			)
		}
		return
	}

	result := runBatch(*target, *concurrency, *bargeInRate)
	printResult(fmt.Sprintf("Benchmark: %s", *target), result)
}

func fmtMs(ms float64) string {
	if ms == 0 {
		return "timeout"
	}
	return fmt.Sprintf("%.1fms", ms)
}

func banner() {
	fmt.Println("=================================================================")
	fmt.Println("    Voice Gateway Benchmark — Go Edition")
	fmt.Println("    KV-Cache Aware | Barge-In | Adaptive TTFT | Indic Languages")
	fmt.Println("=================================================================")
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
