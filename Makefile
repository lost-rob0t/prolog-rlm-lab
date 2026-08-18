MODEL_ID ?= qwen3-8b-q4-k-m
EMACSCLIENT ?= emacsclient
FRONTIER5_MODELS = qwen35-397b-a17b gemini-3-flash grok-45 gpt56-sol claude-opus-5
FRONTIER5_ARGS = $(foreach model,$(FRONTIER5_MODELS),--model-id $(model))

.PHONY: demo doctor probe smoke recursion classes classes-all frontier frontier5 frontier5-dry analyze export \
	emacs-check lab findings report test clean clean-demo

demo:
	./scripts/demo

doctor:
	python3 src/doctor.py --deployment local --model-id $(MODEL_ID)

probe:
	./scripts/bench --preset smoke --deployment local --model-id $(MODEL_ID) --limit 1

smoke:
	./scripts/bench --preset smoke --deployment local --model-id $(MODEL_ID) --quiet
	./scripts/analyze

recursion:
	./scripts/bench --preset recursion --deployment local --model-id $(MODEL_ID) --quiet
	./scripts/analyze

classes:
	./scripts/bench --preset classes --deployment local --model-id $(MODEL_ID) --quiet
	./scripts/analyze

classes-all:
	@echo "Requires a multi-model/router endpoint that advertises every enabled local model."
	./scripts/bench --preset classes --deployment local --quiet
	./scripts/analyze

frontier:
	@echo "Remote benchmark is never implicit. Use make frontier5 for the capped comparison, or invoke scripts/bench explicitly."

frontier5-dry:
	./scripts/bench --preset frontier5 --deployment remote --allow-paid --max-cost-usd 4.25 --dry-run $(FRONTIER5_ARGS)

frontier5:
	./scripts/bench --preset frontier5 --deployment remote --allow-paid --max-cost-usd 4.25 $(FRONTIER5_ARGS)
	./scripts/analyze

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
	python3 -m unittest discover -s tests -v

clean-demo:
	rm -f results/raw/demo-synthetic.jsonl

clean:
	rm -f results/raw/*.jsonl results/raw/*.manifest.json results/summary/*.csv results/summary/*.json results/plots/*.svg public/*.html
	printf '* Results dashboard\n\nNo analyzed benchmark data yet.\n' > results/summary/metrics.org
	printf '* Graphs\n\nNo plots generated yet.\n' > results/summary/plots.org
