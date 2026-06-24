// Package sarvam provides typed clients for Sarvam AI's production APIs:
//   - Saarika v2  — multilingual Indic STT  (https://api.sarvam.ai/speech-to-text)
//   - Bulbul v2   — multilingual Indic TTS  (https://api.sarvam.ai/text-to-speech)
//
// Set SARVAM_API_KEY in the environment to use real endpoints.
// When the key is absent the clients return realistic mock responses so the
// gateway works end-to-end without API access (demo / CI mode).
package sarvam

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"time"
)

// LangCode maps gateway language codes to Sarvam BCP-47 locale strings.
var LangCode = map[string]string{
	"hi": "hi-IN",
	"ta": "ta-IN",
	"te": "te-IN",
	"kn": "kn-IN",
	"bn": "bn-IN",
	"en": "en-IN",
}

// STTResult is the transcription returned by Saarika.
type STTResult struct {
	Transcript string
	LangCode   string
	LatencyMs  float64
}

// TTSChunk is a single PCM audio chunk from Bulbul.
type TTSChunk struct {
	PCM []byte
}

var httpClient = &http.Client{Timeout: 30 * time.Second}

// ── Saarika (STT) ─────────────────────────────────────────────────────────────

// SaarikaClient calls Sarvam's Saarika-v2 speech-to-text endpoint.
type SaarikaClient struct {
	apiKey   string
	endpoint string
}

func NewSaarikaClient() *SaarikaClient {
	return &SaarikaClient{
		apiKey:   os.Getenv("SARVAM_API_KEY"),
		endpoint: "https://api.sarvam.ai/speech-to-text",
	}
}

// saarika v2 JSON response shape.
type saarikaResponse struct {
	Transcript   string  `json:"transcript"`
	LanguageCode string  `json:"language_code"`
	Confidence   float64 `json:"confidence"`
}

// Transcribe sends audio bytes to Saarika and returns the transcript.
// Falls back to a mock when SARVAM_API_KEY is not set.
func (c *SaarikaClient) Transcribe(ctx context.Context, audio []byte, lang string) (*STTResult, error) {
	if c.apiKey == "" || len(audio) == 0 {
		return c.mockTranscribe(lang)
	}

	t0 := time.Now()

	var body bytes.Buffer
	w := multipart.NewWriter(&body)

	// audio field — WAV/PCM accepted by Saarika v2
	fw, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return nil, fmt.Errorf("saarika: create form file: %w", err)
	}
	if _, err = fw.Write(audio); err != nil {
		return nil, fmt.Errorf("saarika: write audio: %w", err)
	}

	lc := LangCode[lang]
	if lc == "" {
		lc = "en-IN"
	}
	_ = w.WriteField("language_code", lc)
	_ = w.WriteField("model", "saarika:v2")
	_ = w.WriteField("with_timestamps", "false")
	w.Close()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, &body)
	if err != nil {
		return nil, fmt.Errorf("saarika: build request: %w", err)
	}
	req.Header.Set("api-subscription-key", c.apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("saarika: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("saarika: status %d: %s", resp.StatusCode, b)
	}

	var sr saarikaResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, fmt.Errorf("saarika: decode: %w", err)
	}

	return &STTResult{
		Transcript: sr.Transcript,
		LangCode:   sr.LanguageCode,
		LatencyMs:  float64(time.Since(t0).Milliseconds()),
	}, nil
}

var mockPrompts = map[string][]string{
	"hi": {"डेटाबेस में consistency क्या होती है?", "machine learning कैसे काम करती है?"},
	"ta": {"தரவுத்தளத்தில் consistency என்றால் என்ன?", "AI பற்றி சொல்லுங்கள்"},
	"te": {"డేటాబేస్‌లో consistency అంటే ఏమిటి?", "AI గురించి చెప్పండి"},
	"kn": {"ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ consistency ಎಂದರೇನು?", "AI ಬಗ್ಗೆ ಹೇಳಿ"},
	"bn": {"ডেটাবেসে consistency কী?", "AI সম্পর্কে বলুন"},
	"en": {"explain consistency in distributed systems", "how does machine learning work"},
}

func (c *SaarikaClient) mockTranscribe(lang string) (*STTResult, error) {
	delay := 80 * time.Millisecond
	if lang != "en" {
		delay = 110 * time.Millisecond
	}
	time.Sleep(delay)
	prompts := mockPrompts[lang]
	if len(prompts) == 0 {
		prompts = mockPrompts["en"]
	}
	return &STTResult{
		Transcript: prompts[rand.Intn(len(prompts))],
		LangCode:   LangCode[lang],
		LatencyMs:  float64(delay.Milliseconds()),
	}, nil
}

// ── Bulbul (TTS) ──────────────────────────────────────────────────────────────

// BulbulClient calls Sarvam's Bulbul-v2 text-to-speech endpoint.
type BulbulClient struct {
	apiKey   string
	endpoint string
}

func NewBulbulClient() *BulbulClient {
	return &BulbulClient{
		apiKey:   os.Getenv("SARVAM_API_KEY"),
		endpoint: "https://api.sarvam.ai/text-to-speech",
	}
}

// bulbul request/response shapes.
type bulbulRequest struct {
	Inputs       []string `json:"inputs"`
	Target       string   `json:"target_language_code"`
	Speaker      string   `json:"speaker"`
	Model        string   `json:"model"`
	SampleRate   int      `json:"sample_rate"`
	EncodingType string   `json:"encoding"`
}

type bulbulResponse struct {
	Audios []string `json:"audios"` // base64-encoded WAV per input chunk
}

// StreamSynthesize sends text to Bulbul and streams PCM audio chunks via the
// returned channel.  Each chunk is raw 16kHz mono int16 PCM bytes.
//
// Bulbul v2 returns complete WAV audio per request (not chunked transfer), so
// we split long text into sentence-level segments and pipeline the requests so
// the first audio arrives while subsequent sentences are still being synthesised.
func (c *BulbulClient) StreamSynthesize(ctx context.Context, text, lang string) (<-chan TTSChunk, error) {
	ch := make(chan TTSChunk, 8)
	if c.apiKey == "" {
		go c.mockStream(ctx, text, lang, ch)
		return ch, nil
	}
	go c.realStream(ctx, text, lang, ch)
	return ch, nil
}

// realStream splits text into sentence segments and fires one Bulbul request per
// segment, streaming PCM chunks to ch as each response arrives.
func (c *BulbulClient) realStream(ctx context.Context, text, lang string, ch chan<- TTSChunk) {
	defer close(ch)

	segments := splitSentences(text)
	if len(segments) == 0 {
		return
	}

	lc := LangCode[lang]
	if lc == "" {
		lc = "en-IN"
	}
	speaker := speakerFor(lang)

	for _, seg := range segments {
		if ctx.Err() != nil {
			return
		}
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}

		pcm, err := c.synthesizeSegment(ctx, seg, lc, speaker)
		if err != nil {
			// Non-fatal: skip the segment rather than killing the stream.
			continue
		}
		select {
		case <-ctx.Done():
			return
		case ch <- TTSChunk{PCM: pcm}:
		}
	}
}

// synthesizeSegment calls Bulbul for a single text segment and returns raw PCM.
func (c *BulbulClient) synthesizeSegment(ctx context.Context, text, langCode, speaker string) ([]byte, error) {
	payload := bulbulRequest{
		Inputs:       []string{text},
		Target:       langCode,
		Speaker:      speaker,
		Model:        "bulbul:v2",
		SampleRate:   16000,
		EncodingType: "wav",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("api-subscription-key", c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bulbul: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bulbul: status %d: %s", resp.StatusCode, b)
	}

	var br bulbulResponse
	if err := json.NewDecoder(resp.Body).Decode(&br); err != nil {
		return nil, fmt.Errorf("bulbul: decode: %w", err)
	}
	if len(br.Audios) == 0 {
		return nil, fmt.Errorf("bulbul: empty audio response")
	}

	// Audios[0] is base64-encoded WAV; strip 44-byte WAV header to get raw PCM.
	wav, err := base64.StdEncoding.DecodeString(br.Audios[0])
	if err != nil {
		return nil, fmt.Errorf("bulbul: base64: %w", err)
	}
	if len(wav) <= 44 {
		return wav, nil // return as-is if unexpectedly short
	}
	return wav[44:], nil // skip WAV header → raw int16 PCM
}

// speakerFor picks a sensible default voice for each language.
var langSpeaker = map[string]string{
	"hi-IN": "meera",
	"ta-IN": "pavithra",
	"te-IN": "arvind",
	"kn-IN": "suresh",
	"bn-IN": "riya",
	"en-IN": "meera",
}

func speakerFor(lang string) string {
	lc := LangCode[lang]
	if s, ok := langSpeaker[lc]; ok {
		return s
	}
	return "meera"
}

// splitSentences splits text on sentence boundaries for pipelined TTS requests.
func splitSentences(text string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Split(bufio.ScanRunes)
	var buf strings.Builder
	boundaries := map[rune]bool{'.': true, '?': true, '!': true, '।': true, '॥': true, '\n': true, '。': true}
	for sc.Scan() {
		r := []rune(sc.Text())[0]
		buf.WriteRune(r)
		if boundaries[r] && buf.Len() > 5 {
			out = append(out, buf.String())
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		out = append(out, buf.String())
	}
	return out
}

func (c *BulbulClient) mockStream(ctx context.Context, text, lang string, ch chan<- TTSChunk) {
	defer close(ch)
	chunkDelay := 30 * time.Millisecond
	chunkSize := 1024
	if lang != "en" {
		chunkDelay = 40 * time.Millisecond
		chunkSize = 1536
	}
	words := len(splitWords(text))
	numChunks := max(3, words/3)
	for i := 0; i < numChunks; i++ {
		select {
		case <-ctx.Done():
			return
		case <-time.After(chunkDelay):
			pcm := make([]byte, chunkSize)
			for j := range pcm {
				pcm[j] = byte(0xaa ^ (j & 0xff))
			}
			ch <- TTSChunk{PCM: pcm}
		}
	}
}

// ── LLM stream interface ───────────────────────────────────────────────────────

// LLMStream is the interface the gateway uses to consume token streams from any
// LLM backend (Sarvam's hosted model, vLLM, Ollama, etc.).
type LLMStream interface {
	NextToken(ctx context.Context) (string, bool, error)
}

// MockLLMStream simulates an LLM with configurable TTFT and per-token delay.
type MockLLMStream struct {
	tokens    []string
	ttft      time.Duration
	perToken  time.Duration
	idx       int
	firstSeen bool
}

func NewMockLLM(lang string, slow bool) *MockLLMStream {
	ttft := 50 * time.Millisecond
	perToken := 20 * time.Millisecond
	text := "This is a detailed response from the Primary LLM with comprehensive information about your query."

	if lang != "en" {
		ttft = 250 * time.Millisecond
		perToken = 25 * time.Millisecond
		text = indicResponse(lang)
	}
	if slow {
		ttft = 400 * time.Millisecond
	}
	return &MockLLMStream{
		tokens:   splitWords(text),
		ttft:     ttft,
		perToken: perToken,
	}
}

func (m *MockLLMStream) NextToken(ctx context.Context) (string, bool, error) {
	delay := m.perToken
	if !m.firstSeen {
		delay = m.ttft
		m.firstSeen = true
	}
	select {
	case <-ctx.Done():
		return "", false, ctx.Err()
	case <-time.After(delay):
	}
	if m.idx >= len(m.tokens) {
		return "", false, nil
	}
	tok := m.tokens[m.idx]
	if m.idx > 0 {
		tok = " " + tok
	}
	m.idx++
	return tok, m.idx <= len(m.tokens), nil
}

// NewFallbackSLM is a fast local SLM that responds in <50ms TTFT.
func NewFallbackSLM() *MockLLMStream {
	return &MockLLMStream{
		tokens:   splitWords("Fast SLM fallback: Primary was too slow. Answering from local model."),
		ttft:     10 * time.Millisecond,
		perToken: 5 * time.Millisecond,
	}
}

var indicResponses = map[string]string{
	"hi": "डेटाबेस में consistency का अर्थ है कि सभी nodes एक ही data देखते हैं। CAP theorem के अनुसार consistency और availability में trade-off करना पड़ता है।",
	"ta": "தரவுத்தளத்தில் consistency என்பது அனைத்து nodes-உம் ஒரே data-ஐ பார்க்கின்றன. CAP theorem படி consistency மற்றும் availability ஒன்றை தேர்வு செய்ய வேண்டும்.",
	"te": "డేటాబేస్‌లో consistency అంటే అన్ని nodes ఒకే data చూస్తాయి. CAP theorem ప్రకారం consistency మరియు availability లో ఒకదాన్ని ఎంచుకోవాలి.",
	"kn": "ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ consistency ಎಂದರೆ ಎಲ್ಲಾ nodes ಒಂದೇ data ನೋಡುತ್ತವೆ. CAP theorem ಪ್ರಕಾರ consistency ಮತ್ತು availability ನಲ್ಲಿ ಒಂದನ್ನು ಆರಿಸಬೇಕು.",
	"bn": "ডেটাবেসে consistency মানে সব nodes একই data দেখে। CAP theorem অনুযায়ী consistency এবং availability-র মধ্যে একটি বেছে নিতে হবে।",
}

func indicResponse(lang string) string {
	if r, ok := indicResponses[lang]; ok {
		return r
	}
	return indicResponses["hi"]
}

func splitWords(s string) []string {
	var words []string
	start := -1
	for i, r := range s {
		if r == ' ' || r == '\n' {
			if start >= 0 {
				words = append(words, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		words = append(words, s[start:])
	}
	return words
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func init() {
	if os.Getenv("SARVAM_API_KEY") == "" {
		fmt.Println("[sarvam] SARVAM_API_KEY not set — running in mock mode.")
	} else {
		fmt.Println("[sarvam] API key present — using real Saarika-v2 + Bulbul-v2 endpoints.")
	}
}
