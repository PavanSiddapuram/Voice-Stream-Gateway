// Package ws implements the WebSocket connection handler.
//
// Each connection runs in a dedicated goroutine (readPump). The STT→LLM→TTS
// pipeline runs as a child goroutine with a cancellable context — barge-in
// simply calls cancel(), which is instant and contention-free compared to
// Python asyncio.Task.cancel() which can silently fail under event loop load.
//
// Goroutine budget per connection: 2 (readPump + pipeline when active).
// Memory per idle connection: ~8KB (goroutine stack) + session struct.
// Python asyncio equivalent: single event loop shared across ALL connections.
package ws

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"voice-gateway/audio"
	"voice-gateway/ring"
	"voice-gateway/router"
	"voice-gateway/sarvam"
	"voice-gateway/session"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

type msg struct {
	Event string `json:"event"`
	Data  string `json:"data"`
}

type speechEndPayload struct {
	Event    string `json:"event"`
	Prompt   string `json:"prompt"`
	Language string `json:"language"`
}

// Handler is the shared gateway state wired into each WebSocket connection.
type Handler struct {
	Ring    *ring.HashRing
	Store   *session.Store
	Tracker *router.TTFTTracker
	Metrics *router.Metrics
	Saarika *sarvam.SaarikaClient
	Bulbul  *sarvam.BulbulClient

	// SlowPrimary is toggled by /toggle-latency to simulate degraded primary LLM.
	SlowPrimary bool
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws upgrade error: %v", err)
		return
	}
	defer conn.Close()

	sessionID := r.URL.Query().Get("session_id")
	if sessionID == "" {
		sessionID = fmt.Sprintf("session-%d", time.Now().UnixMilli())
	}

	worker := h.Ring.GetNode(sessionID)
	sess := h.Store.GetOrCreate(sessionID, worker)

	sess.Lock()
	sess.Connected = true
	sess.Unlock()

	log.Printf("[ws] connected: %s → %s (turns=%d)", sessionID, worker, sess.TurnCount)

	send := func(event, data string) {
		conn.WriteJSON(msg{Event: event, Data: data})
	}

	// Announce worker assignment and resume status to the client.
	initData, _ := json.Marshal(map[string]any{
		"worker":        worker,
		"session_id":    sessionID,
		"resumed":       sess.TurnCount > 0,
		"history_turns": sess.TurnCount,
		"language":      sess.Language,
	})
	send("session_init", string(initData))

	vad := audio.NewVAD()

	defer func() {
		sess.Lock()
		sess.Connected = false
		sess.State = session.StateListening
		sess.Unlock()
		log.Printf("[ws] disconnected: %s (history preserved, turns=%d)", sessionID, sess.TurnCount)
	}()

	for {
		msgType, payload, err := conn.ReadMessage()
		if err != nil {
			break
		}

		switch msgType {
		case websocket.BinaryMessage:
			// PCM audio frame from microphone.
			// Run VAD; if pipeline is active and user is speaking → barge-in.
			processed := audio.Resample(payload, 48000, 16000)
			vadResult := vad.Process(processed)

			sess.Lock()
			state := sess.State
			sess.Unlock()

			if state != session.StateListening && vadResult.IsSpeech && vadResult.SpeechFrames > 3 {
				desc := sess.HandleBargeIn()
				if desc != "" {
					log.Printf("[barge-in] %s: %s", sessionID, desc)
					send("barge_in_triggered", desc)
				}
			}

		case websocket.TextMessage:
			var envelope map[string]string
			if err := json.Unmarshal(payload, &envelope); err != nil {
				continue
			}
			event := envelope["event"]

			switch event {
			case "barge_in", "speech_start":
				desc := sess.HandleBargeIn()
				if desc != "" {
					log.Printf("[barge-in ctrl] %s: %s", sessionID, desc)
					send("barge_in_triggered", desc)
				}

			case "speech_end":
				var p speechEndPayload
				json.Unmarshal(payload, &p)
				langHint := p.Language
				if _, ok := router.LangNames[langHint]; !ok {
					langHint = "auto"
				}
				sess.ResetPipeline()
				ctx, cancel := context.WithCancel(context.Background())
				sess.Lock()
				sess.CancelPipeline = cancel
				sess.Unlock()
				go h.runPipeline(ctx, conn, sess, send, p.Prompt, langHint)
			}
		}
	}
}

// runPipeline executes the STT → LLM → TTS pipeline for one turn.
// It is launched as a goroutine and cancellable via ctx for barge-in.
func (h *Handler) runPipeline(
	ctx context.Context,
	conn *websocket.Conn,
	sess *session.Session,
	send func(string, string),
	promptOverride, langHint string,
) {
	sess.Lock()
	sess.State = session.StateThinking
	sess.Unlock()

	send("status", "Transcribing Speech...")

	// --- 1. STT ---
	prompt := promptOverride
	if prompt == "" {
		effectiveLang := langHint
		if !router.IndicLangs[effectiveLang] {
			effectiveLang = "en"
		}
		result, err := h.Saarika.Transcribe(ctx, nil, effectiveLang)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			send("status", "STT error: "+err.Error())
			return
		}
		prompt = result.Transcript
	}

	// --- 2. Language detection ---
	lang := langHint
	if lang == "auto" || lang == "" {
		lang = router.DetectLanguage(prompt)
	}

	sess.Lock()
	sess.Language = lang
	sess.History = append(sess.History, session.Turn{Role: "user", Content: prompt})
	sess.Unlock()

	h.Metrics.IncPrimary(lang) // count turn (will be corrected if fallback occurs)

	send("transcription", prompt)
	langData, _ := json.Marshal(map[string]any{
		"lang":                lang,
		"name":                router.LangNames[lang],
		"adaptive_timeout_ms": h.Tracker.Timeout(lang).Milliseconds(),
	})
	send("language_detected", string(langData))

	// --- 3. LLM with adaptive TTFT fallback ---
	send("status", fmt.Sprintf("Querying %s LLM...", router.LangNames[lang]))

	timeout := h.Tracker.Timeout(lang)
	primary := sarvam.NewMockLLM(lang, h.SlowPrimary)
	fallback := sarvam.NewFallbackSLM()

	ttftStart := time.Now()
	firstTok, hasMore, err := primary.NextToken(ctx)
	ttft := time.Since(ttftStart)

	var activeStream sarvam.LLMStream
	usedFallback := false

	if err != nil || !hasMore && firstTok == "" || ttft >= timeout {
		log.Printf("[route] [%s] primary slow/failed (%.0fms > %.0fms), falling back",
			strings.ToUpper(lang), ttft.Seconds()*1000, timeout.Seconds()*1000)
		h.Metrics.IncFallback(lang)
		usedFallback = true
		activeStream = fallback
		// Re-consume the first token from fallback
		firstTok, hasMore, err = fallback.NextToken(ctx)
	} else {
		h.Tracker.Record(lang, ttft)
		log.Printf("[route] [%s] primary TTFT=%.0fms (limit=%.0fms)",
			strings.ToUpper(lang), ttft.Seconds()*1000, timeout.Seconds()*1000)
		activeStream = primary
	}

	if err != nil {
		return
	}
	_ = usedFallback

	// --- 4. Sentence-chunked TTS streaming ---
	sess.Lock()
	sess.State = session.StateSpeaking
	sess.Unlock()

	send("status", "Streaming Audio Out...")

	chunker := &sentenceChunker{maxTokens: 15}
	allTokens := []string{}

	flushChunk := func(text string) {
		if ctx.Err() != nil {
			return
		}
		ttsCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		ch, err := h.Bulbul.StreamSynthesize(ttsCtx, text, lang)
		if err != nil {
			return
		}
		for chunk := range ch {
			if ctx.Err() != nil {
				return
			}
			conn.WriteMessage(websocket.BinaryMessage, chunk.PCM)
			sess.Lock()
			sess.SentTokenCount = len(sess.GeneratedTokens)
			sess.Unlock()
		}
	}

	// Process first token.
	allTokens = append(allTokens, firstTok)
	sess.Lock()
	sess.GeneratedTokens = append(sess.GeneratedTokens, firstTok)
	sess.Unlock()
	if chunk := chunker.push(firstTok); chunk != "" {
		flushChunk(chunk)
	}

	// Stream remaining tokens.
	for hasMore {
		tok, more, err := activeStream.NextToken(ctx)
		if err != nil || ctx.Err() != nil {
			return
		}
		hasMore = more
		if tok == "" {
			continue
		}
		allTokens = append(allTokens, tok)
		sess.Lock()
		sess.GeneratedTokens = append(sess.GeneratedTokens, tok)
		sess.Unlock()
		if chunk := chunker.push(tok); chunk != "" {
			flushChunk(chunk)
		}
	}
	if remaining := chunker.flush(); remaining != "" {
		flushChunk(remaining)
	}

	fullResponse := strings.Join(allTokens, "")
	sess.Lock()
	sess.History = append(sess.History, session.Turn{Role: "assistant", Content: fullResponse})
	sess.TurnCount++
	sess.State = session.StateListening
	sess.GeneratedTokens = nil
	sess.SentTokenCount = 0
	sess.Unlock()

	send("status", "Finished speaking.")
	send("assistant_response", fullResponse)
	log.Printf("[pipeline] %s turn %d complete (lang=%s)", sess.ID, sess.TurnCount, lang)
}

// sentenceChunker buffers tokens and flushes on sentence boundaries or every
// maxTokens tokens, enabling TTS to start before the LLM finishes responding.
type sentenceChunker struct {
	maxTokens int
	buf       []string
}

var boundaries = "。.?!।॥\n"

func (c *sentenceChunker) push(token string) string {
	c.buf = append(c.buf, token)
	text := strings.Join(c.buf, "")
	if strings.ContainsAny(text, boundaries) || len(c.buf) >= c.maxTokens {
		c.buf = nil
		return strings.TrimSpace(text)
	}
	return ""
}

func (c *sentenceChunker) flush() string {
	if len(c.buf) == 0 {
		return ""
	}
	text := strings.TrimSpace(strings.Join(c.buf, ""))
	c.buf = nil
	return text
}
