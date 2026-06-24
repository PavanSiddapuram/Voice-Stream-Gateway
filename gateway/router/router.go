// Package router implements adaptive TTFT (Time-To-First-Token) tracking and
// per-language LLM fallback decisions.
//
// Key insight for Indic languages: tokenization is 3-4x less efficient than
// English (e.g. Hindi text produces 3x more tokens than equivalent English),
// so a fixed 200ms fallback threshold causes constant false-positive fallbacks.
// This router tracks a rolling p90 per language and widens the window adaptively.
package router

import (
	"math"
	"sort"
	"sync"
	"time"
)

// Language codes supported by the gateway.
const (
	LangEN = "en"
	LangHI = "hi"
	LangTA = "ta"
	LangTE = "te"
	LangKN = "kn"
	LangBN = "bn"
)

// IndicLangs is the set of Indic language codes.
var IndicLangs = map[string]bool{
	LangHI: true,
	LangTA: true,
	LangTE: true,
	LangKN: true,
	LangBN: true,
}

// LangNames maps ISO codes to display names.
var LangNames = map[string]string{
	LangEN: "English",
	LangHI: "Hindi (हिंदी)",
	LangTA: "Tamil (தமிழ்)",
	LangTE: "Telugu (తెలుగు)",
	LangKN: "Kannada (ಕನ್ನಡ)",
	LangBN: "Bengali (বাংলা)",
}

// baseTTFT is the starting fallback window per language family.
// Indic gets 350ms because Indic-tuned LLMs have higher tokenization overhead.
var baseTTFT = map[bool]time.Duration{
	false: 200 * time.Millisecond, // English
	true:  350 * time.Millisecond, // Indic
}

const maxSamples = 50

// TTFTTracker maintains a per-language rolling window of observed first-token
// latencies and derives an adaptive timeout from the p90 of recent samples.
type TTFTTracker struct {
	mu      sync.RWMutex
	samples map[string][]float64 // seconds
}

func NewTTFTTracker() *TTFTTracker {
	return &TTFTTracker{samples: make(map[string][]float64)}
}

// Record adds an observed TTFT sample for the given language.
func (t *TTFTTracker) Record(lang string, elapsed time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.samples[lang]
	s = append(s, elapsed.Seconds())
	if len(s) > maxSamples {
		s = s[1:]
	}
	t.samples[lang] = s
}

// Timeout returns the adaptive TTFT threshold for lang.
// With fewer than 5 samples it returns the base; afterwards it uses p90 × 1.5.
func (t *TTFTTracker) Timeout(lang string) time.Duration {
	base := baseTTFT[IndicLangs[lang]]
	t.mu.RLock()
	s := append([]float64{}, t.samples[lang]...)
	t.mu.RUnlock()
	if len(s) < 5 {
		return base
	}
	p90 := percentile(s, 90)
	adaptive := time.Duration(p90*1.5*1e9) * time.Nanosecond
	if adaptive > base {
		return adaptive
	}
	return base
}

// Stats returns a snapshot of per-language TTFT statistics for /health.
func (t *TTFTTracker) Stats() map[string]LangStats {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]LangStats, len(t.samples))
	for lang, s := range t.samples {
		if len(s) == 0 {
			continue
		}
		out[lang] = LangStats{
			P50Ms:   math.Round(percentile(s, 50)*1000*10) / 10,
			P90Ms:   math.Round(percentile(s, 90)*1000*10) / 10,
			Samples: len(s),
		}
	}
	return out
}

type LangStats struct {
	P50Ms   float64 `json:"p50_ms"`
	P90Ms   float64 `json:"p90_ms"`
	Samples int     `json:"samples"`
}

// DetectLanguage returns the ISO language code for text by scanning Unicode
// codepoints. Falls back to "en" if no Indic script is found.
func DetectLanguage(text string) string {
	for _, r := range text {
		cp := int(r)
		switch {
		case cp >= 0x0900 && cp <= 0x097F:
			return LangHI // Devanagari
		case cp >= 0x0980 && cp <= 0x09FF:
			return LangBN // Bengali
		case cp >= 0x0B80 && cp <= 0x0BFF:
			return LangTA // Tamil
		case cp >= 0x0C00 && cp <= 0x0C7F:
			return LangTE // Telugu
		case cp >= 0x0C80 && cp <= 0x0CFF:
			return LangKN // Kannada
		}
	}
	return LangEN
}

// Metrics holds gateway-wide routing counters.
type Metrics struct {
	mu               sync.Mutex
	PrimaryHits      int
	FallbackHits     int
	LangDistribution map[string]int
}

func NewMetrics() *Metrics {
	return &Metrics{LangDistribution: make(map[string]int)}
}

func (m *Metrics) IncPrimary(lang string) {
	m.mu.Lock()
	m.PrimaryHits++
	m.LangDistribution[lang]++
	m.mu.Unlock()
}

func (m *Metrics) IncFallback(lang string) {
	m.mu.Lock()
	m.FallbackHits++
	m.LangDistribution[lang]++
	m.mu.Unlock()
}

func (m *Metrics) Snapshot() (primary, fallback int, dist map[string]int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make(map[string]int, len(m.LangDistribution))
	for k, v := range m.LangDistribution {
		cp[k] = v
	}
	return m.PrimaryHits, m.FallbackHits, cp
}

func percentile(sorted []float64, p float64) float64 {
	s := append([]float64{}, sorted...)
	sort.Float64s(s)
	if len(s) == 0 {
		return 0
	}
	idx := p / 100.0 * float64(len(s)-1)
	lo := int(idx)
	hi := lo + 1
	if hi >= len(s) {
		return s[lo]
	}
	return s[lo] + (idx-float64(lo))*(s[hi]-s[lo])
}
