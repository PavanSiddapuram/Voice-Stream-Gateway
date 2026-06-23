// Package session manages voice session state across WebSocket reconnects.
// History and worker assignment persist for SESSION_TTL; a background goroutine
// evicts idle sessions so memory is bounded without a GC sweep on every request.
package session

import (
	"context"
	"sync"
	"time"
)

const SESSION_TTL = time.Hour

type State string

const (
	StateListening State = "Listening"
	StateThinking  State = "Thinking"
	StateSpeaking  State = "Speaking"
)

type Turn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Session struct {
	mu        sync.Mutex
	ID        string
	Worker    string
	Language  string
	State     State
	History   []Turn
	TurnCount int
	Connected bool
	CreatedAt time.Time
	LastUsed  time.Time

	// CancelPipeline cancels the currently running STT→LLM→TTS goroutine.
	// Replaced atomically on each new pipeline start.
	CancelPipeline context.CancelFunc

	// GeneratedTokens tracks tokens produced in the current pipeline turn
	// so barge-in can truncate history to exactly what was spoken.
	GeneratedTokens []string
	SentTokenCount  int
}

func (s *Session) Lock()   { s.mu.Lock() }
func (s *Session) Unlock() { s.mu.Unlock() }

// HandleBargeIn cancels the active pipeline and truncates history.
// Returns a human-readable description of what was interrupted, or "" if idle.
func (s *Session) HandleBargeIn() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.State == StateListening {
		return ""
	}
	prev := s.State
	if s.CancelPipeline != nil {
		s.CancelPipeline()
	}
	spoken := ""
	if len(s.GeneratedTokens) > 0 {
		end := s.SentTokenCount
		if end > len(s.GeneratedTokens) {
			end = len(s.GeneratedTokens)
		}
		for _, t := range s.GeneratedTokens[:end] {
			spoken += t
		}
		if len(s.History) > 0 && s.History[len(s.History)-1].Role == "assistant" {
			s.History[len(s.History)-1].Content = spoken + " [INTERRUPTED]"
		} else if spoken != "" {
			s.History = append(s.History, Turn{Role: "assistant", Content: spoken + " [INTERRUPTED]"})
		}
	}
	s.GeneratedTokens = nil
	s.SentTokenCount = 0
	s.State = StateListening
	return string(prev) + " interrupted. Spoken: '" + spoken + "'"
}

func (s *Session) ResetPipeline() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.CancelPipeline != nil {
		s.CancelPipeline()
	}
	s.GeneratedTokens = nil
	s.SentTokenCount = 0
}

// Store is a thread-safe session registry backed by sync.Map.
type Store struct {
	m sync.Map
}

func NewStore() *Store {
	return &Store{}
}

func (st *Store) GetOrCreate(id, worker string) *Session {
	now := time.Now()
	v, loaded := st.m.LoadOrStore(id, &Session{
		ID:        id,
		Worker:    worker,
		Language:  "en",
		State:     StateListening,
		CreatedAt: now,
		LastUsed:  now,
	})
	s := v.(*Session)
	if loaded {
		s.mu.Lock()
		s.LastUsed = now
		s.mu.Unlock()
	}
	return s
}

func (st *Store) Get(id string) (*Session, bool) {
	v, ok := st.m.Load(id)
	if !ok {
		return nil, false
	}
	return v.(*Session), true
}

func (st *Store) Delete(id string) {
	st.m.Delete(id)
}

// Snapshot returns a point-in-time slice of all sessions (for /sessions endpoint).
func (st *Store) Snapshot() []*Session {
	var out []*Session
	st.m.Range(func(_, v any) bool {
		out = append(out, v.(*Session))
		return true
	})
	return out
}

// StartEviction runs a background goroutine that removes sessions idle longer
// than SESSION_TTL. Call once at startup.
func (st *Store) StartEviction(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(10 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				cutoff := time.Now().Add(-SESSION_TTL)
				st.m.Range(func(k, v any) bool {
					s := v.(*Session)
					s.mu.Lock()
					idle := !s.Connected && s.LastUsed.Before(cutoff)
					s.mu.Unlock()
					if idle {
						st.m.Delete(k)
					}
					return true
				})
			}
		}
	}()
}

// ActiveCount returns the number of currently connected sessions.
func (st *Store) ActiveCount() int {
	n := 0
	st.m.Range(func(_, v any) bool {
		if v.(*Session).Connected {
			n++
		}
		return true
	})
	return n
}

// TotalCount returns total sessions in store.
func (st *Store) TotalCount() int {
	n := 0
	st.m.Range(func(_, _ any) bool { n++; return true })
	return n
}
