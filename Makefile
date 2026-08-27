MODEL_ID ?= qwen3-8b-q4-k-m
EMACSCLIENT ?= emacsclient
RLM_LAB ?= bash ./rlm-lab
FRONTIER5_MODELS = qwen35-397b-a17b gemini-3-flash grok-45 gpt56-sol claude-opus-5
FRONTIER5_ARGS = $(foreach model,$(FRONTIER5_MODELS),--model-id $(model))
OPENROUTER_FREE_MODELS ?= 5
OPENROUTER_FREE_REQUESTS ?= 40
SWE_LIMIT ?= 3
SWE_TURNS ?= 8
SWE_SESSION ?=
SWE_WORKERS ?= 2

.PHONY: demo doctor probe smoke recursion compare classes classes-all starintel frontier frontier5 frontier5-dry \
	openrouter-free-list openrouter-free-deep openrouter-free-router \
	swebench-deps swebench-doctor swebench-gold swebench-generate swebench-evaluate \
	rlm-doctor rlm-smoke rlm-simple analyze export emacs-check lab findings report test clean clean-demo

demo:
	./scripts/demo

doctor:
	python3 src/doctor.py --deployment local --model-id $(MODEL_ID)

probe:
	./scripts/bench --preset smoke --deployment local --model-id $(MODEL_ID) --limit 1

smoke:
	./scripts/bench --preset smoke --deployment local --model-id $(MODEL_ID) --quiet || test $$? -eq 1
	./scripts/analyze

# Production prolog-rlm runtime lane. This is deliberately separate from the
# historical Python benchmark harness above: no duplicate RLM implementation.
rlm-doctor:
	$(RLM_LAB) doctor

# Credential-free production runtime smoke; suitable for local use and CI.
rlm-smoke:
	$(RLM_LAB) smoke

# Live depth-1 RLM against the configured local/OpenAI-compatible endpoint.
rlm-simple:
	$(RLM_LAB) simple

# Explicit causal comparison: base model -> plain agent -> Prolog-RLM + verifier.
compare:
	./scripts/bench --preset compare --deployment local --model-id $(MODEL_ID) --quiet || test $$? -eq 1
	./scripts/analyze

# Legacy recursion sweep retained for sessions already collected with old labels.
recursion:
	./scripts/bench --preset recursion --deployment local --model-id $(MODEL_ID) --quiet || test $$? -eq 1
	./scripts/analyze

classes:
	./scripts/bench --preset classes --deployment local --model-id $(MODEL_ID) --quiet || test $$? -eq 1
	./scripts/analyze

classes-all:
	@echo "Requires a multi-model/router endpoint that advertises every enabled local model."
	./scripts/bench --preset classes --deployment local --quiet || test $$? -eq 1
	./scripts/analyze

starintel:
	./scripts/bench --preset starintel --deployment local --model-id $(MODEL_ID) --quiet || test $$? -eq 1
	./scripts/analyze

frontier:
	@echo "Remote benchmark is never implicit. Use make frontier5 for the capped comparison, or invoke scripts/bench explicitly."

frontier5-dry:
	./scripts/bench --preset frontier5 --deployment remote --allow-paid --max-cost-usd 4.25 --dry-run $(FRONTIER5_ARGS)

frontier5:
	./scripts/bench --preset frontier5 --deployment remote --allow-paid --max-cost-usd 4.25 $(FRONTIER5_ARGS) || test $$? -eq 1
	./scripts/analyze

# Dynamic OpenRouter free-model lane. list makes no inference request; deep is
# capped by request count and aborts if provider-reported cost is nonzero.
openrouter-free-list:
	python3 src/openrouter_free.py list

openrouter-free-deep:
	python3 src/openrouter_free.py deep --max-models $(OPENROUTER_FREE_MODELS) --max-requests $(OPENROUTER_FREE_REQUESTS)

openrouter-free-router:
	python3 src/openrouter_free.py deep --max-models $(OPENROUTER_FREE_MODELS) --max-requests $(OPENROUTER_FREE_REQUESTS) --include-router

# SWE-bench is optional because its Docker/dataset dependencies are much larger
# than the core symbolic lab.
swebench-deps:
	python3 -m pip install -r requirements-swebench.txt

swebench-doctor:
	python3 src/swebench_lab.py doctor

# Validate the official Docker scorer with a known gold patch before measuring models.
swebench-gold:
	python3 -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite --split test --predictions_path gold --max_workers 1 --instance_ids sympy__sympy-20590 --run_id prolog-rlm-lab-gold

swebench-generate:
	python3 src/swebench_lab.py generate --model-id $(MODEL_ID) --limit $(SWE_LIMIT) --turns $(SWE_TURNS)

swebench-evaluate:
	@test -n "$(SWE_SESSION)" || (echo "set SWE_SESSION=<session id from swebench-generate>"; exit 2)
	python3 src/swebench_lab.py evaluate --session-id $(SWE_SESSION) --workers $(SWE_WORKERS)

analyze:
	./scripts/analyze

export:
	./scripts/export

emacs-check:
	@$(EMACSCLIENT) --eval '(list :emacs emacs-version :gptel (and (require (quote gptel-request) nil t) t) :ob-async (and (require (quote ob-async) nil t) t))'

lab:
	$(EMACSCLIENT) -n "$(CURDIR)/LAB.org"

findings:
	$(EMACSCLIENT) --eval '(progn (load-file "$(CURDIR)/emacs/rlm-lab.el") (rlm-lab-generate-findings) "FINDINGS.org generation dispatched")'

report:
	$(EMACSCLIENT) --eval '(progn (load-file "$(CURDIR)/emacs/rlm-lab.el") (rlm-lab-generate-findings-and-export) "FINDINGS.org generation and HTML export dispatched")'

test:
	bash -n ./rlm-lab
	bash ./rlm-lab help >/dev/null
	python3 -m unittest discover -s tests -v

clean-demo:
	rm -f results/raw/demo-synthetic.jsonl

clean:
	rm -f results/raw/*.jsonl results/raw/*.manifest.json results/summary/*.csv results/summary/*.json results/plots/*.svg public/*.html
	rm -rf results/free results/swebench
	printf '* Results dashboard\n\nNo analyzed benchmark data yet.\n' > results/summary/metrics.org
	printf '* Graphs\n\nNo plots generated yet.\n' > results/summary/plots.org
