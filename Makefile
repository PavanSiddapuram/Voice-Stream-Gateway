.PHONY: build run-python run-go run-all bench-python bench-go bench-compare \
        bench-scale test-leaks demo stop

PYTHON_PORT  := 8000
GO_PORT      := 8080
GO_BIN       := /tmp/voice-gateway

# ── Build ──────────────────────────────────────────────────────────────────────

build:
	@echo "[build] Compiling Go gateway..."
	@cd gateway && go build -o $(GO_BIN) .
	@echo "[build] Installing Python deps..."
	@pip install -r requirements.txt -q
	@echo "[build] Done."

# ── Run ────────────────────────────────────────────────────────────────────────

run-python:
	python -m uvicorn main:app --host 127.0.0.1 --port $(PYTHON_PORT)

run-go: build
	$(GO_BIN) --port $(GO_PORT)

# Run both servers in background, open dashboard in browser.
run-all: build
	@echo "[run-all] Starting Python inference server on :$(PYTHON_PORT) ..."
	python -m uvicorn main:app --host 127.0.0.1 --port $(PYTHON_PORT) > /tmp/py-server.log 2>&1 &
	@sleep 2
	@echo "[run-all] Starting Go gateway on :$(GO_PORT) ..."
	$(GO_BIN) --port $(GO_PORT) > /tmp/go-server.log 2>&1 &
	@sleep 2
	@echo "[run-all] Both servers up."
	@echo "  Python dashboard : http://127.0.0.1:$(PYTHON_PORT)"
	@echo "  Go health        : http://127.0.0.1:$(GO_PORT)/health"
	@echo "  Go sessions      : http://127.0.0.1:$(GO_PORT)/sessions"
	@echo ""
	@echo "  Run 'make bench-compare' to see the side-by-side benchmark."

stop:
	@kill $$(lsof -ti:$(PYTHON_PORT),$(GO_PORT)) 2>/dev/null || true
	@echo "[stop] Servers stopped."

# ── Benchmarks ─────────────────────────────────────────────────────────────────

# Python gateway — p50/p90/p99 at 50 concurrent sessions with 20% barge-in rate
bench-python:
	python tests/pressure_test.py --concurrency 50 --barge-in-rate 0.2 --server 127.0.0.1:$(PYTHON_PORT)

# Go gateway — same workload
bench-go:
	cd gateway/benchmark && go run bench.go \
		--target ws://127.0.0.1:$(GO_PORT)/ws \
		--concurrency 50 \
		--barge-in-rate 0.2

# Full scaling comparison: Go vs Python at 50 / 500 / 2000 sessions.
# Both servers must be running (make run-all) before calling this.
bench-compare:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  Voice Gateway Benchmark: Go vs Python"
	@echo "  KV-Cache Routing | Barge-In | Adaptive TTFT | Indic Languages"
	@echo "═══════════════════════════════════════════════════════════════════"
	@$(MAKE) _bench-level N=50
	@$(MAKE) _bench-level N=500
	@$(MAKE) _bench-level N=2000
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"

# Internal target — called by bench-compare with N set externally.
_bench-level:
	@echo ""
	@echo "────── $(N) concurrent sessions ──────"
	@echo "[Python :$(PYTHON_PORT)]"
	@python tests/pressure_test.py --concurrency $(N) --barge-in-rate 0.2 \
		--server 127.0.0.1:$(PYTHON_PORT) 2>/dev/null \
		| grep -E "Throughput|Handshake|TTFT|Barge-In"
	@echo ""
	@echo "[Go     :$(GO_PORT)]"
	@cd gateway/benchmark && go run bench.go \
		--target ws://127.0.0.1:$(GO_PORT)/ws \
		--concurrency $(N) --barge-in-rate 0.2 2>/dev/null \
		| grep -E "Throughput|Handshake|TTFT|Barge"

# Scale stress test: Go only, 10 000 concurrent sessions
bench-scale:
	cd gateway/benchmark && go run bench.go \
		--target ws://127.0.0.1:$(GO_PORT)/ws \
		--concurrency 10000 \
		--barge-in-rate 0.2

# ── Quality ────────────────────────────────────────────────────────────────────

# Verify zero session leaks after 50 connect/disconnect cycles
test-leaks:
	python tests/efficiency_test.py --cycles 50 --server 127.0.0.1:$(PYTHON_PORT)

# Full demo sequence: build → start → benchmark → stop
demo: build run-all
	@sleep 3
	$(MAKE) bench-compare
	$(MAKE) stop
