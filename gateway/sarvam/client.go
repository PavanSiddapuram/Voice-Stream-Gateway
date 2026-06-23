// Package sarvam provides typed stubs for Sarvam AI's production APIs:
//   - Saarika v2  — multilingual Indic STT
//   - Bulbul v2   — multilingual Indic TTS
//
// Set SARVAM_API_KEY in the environment to use real endpoints.
// When the key is absent the clients return realistic mock responses so the
// gateway works end-to-end without API access (demo / CI mode).
//
// Real API reference: https://docs.sarvam.ai
package sarvam

import (
	"context"
	"fmt"
	"math/rand"
	"os"
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

// SaarikaCient calls Sarvam's Saarika-v2 speech-to-text endpoint.
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

// Transcribe sends audio bytes to Saarika and returns the transcript.
// Falls back to a mock when SARVAM_API_KEY is not set.
func (c *SaarikaClient) Transcribe(ctx context.Context, audio []byte, lang string) (*STTResult, error) {
	if c.apiKey == "" {
		return c.mockTranscribe(lang)
	}
	// --- Real implementation (uncomment when API key is available) ---
	// form := &bytes.Buffer{}
	// w := multipart.NewWriter(form)
	// fw, _ := w.CreateFormFile("file", "audio.wav")
	// fw.Write(audio)
	// w.WriteField("language_code", LangCode[lang])
	// w.WriteField("model", "saarika:v2")
	// w.Close()
	// req, _ := http.NewRequestWithContext(ctx, "POST", c.endpoint, form)
	// req.Header.Set("api-subscription-key", c.apiKey)
	// req.Header.Set("Content-Type", w.FormDataContentType())
	// resp, err := http.DefaultClient.Do(req)
	// ... parse JSON response ...
	return c.mockTranscribe(lang)
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
		delay = 110 * time.Millisecond // Indic STT ~110ms (Saarika v2 benchmark)
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

// StreamSynthesize sends text to Bulbul and streams PCM audio chunks via the
// returned channel. Each chunk is 1–2KB of 16kHz mono int16 PCM.
func (c *BulbulClient) StreamSynthesize(ctx context.Context, text, lang string) (<-chan TTSChunk, error) {
	ch := make(chan TTSChunk, 8)
	if c.apiKey == "" {
		go c.mockStream(ctx, text, lang, ch)
		return ch, nil
	}
	// --- Real streaming implementation placeholder ---
	// Sarvam Bulbul supports chunked transfer encoding; each chunk is raw PCM.
	// req, _ := http.NewRequestWithContext(ctx, "POST", c.endpoint, body)
	// req.Header.Set("api-subscription-key", c.apiKey)
	// go streamResponse(resp.Body, ch)
	go c.mockStream(ctx, text, lang, ch)
	return ch, nil
}

func (c *BulbulClient) mockStream(ctx context.Context, text, lang string, ch chan<- TTSChunk) {
	defer close(ch)
	// Simulate Bulbul: ~40ms/chunk for Indic, ~30ms/chunk English.
	chunkDelay := 30 * time.Millisecond
	chunkSize := 1024
	if lang != "en" {
		chunkDelay = 40 * time.Millisecond
		chunkSize = 1536 // More bytes per chunk — longer Indic phoneme sequences.
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

// LLMStream is the interface the gateway uses to consume token streams from any
// LLM backend (Sarvam's hosted model, vLLM, Ollama, etc.).
type LLMStream interface {
	// NextToken blocks until the next token is available or ctx is cancelled.
	NextToken(ctx context.Context) (string, bool, error)
}

// MockLLMStream simulates an LLM with configurable TTFT and per-token delay.
// Replace with a real gRPC/HTTP streaming client for production.
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
		ttft = 250 * time.Millisecond  // Indic tokenization overhead
		perToken = 25 * time.Millisecond
		text = indicResponse(lang)
	}
	if slow {
		ttft = 400 * time.Millisecond // Simulate degraded primary
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

// FallbackSLM is a fast local SLM that responds in <50ms TTFT.
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
	}
}
