MODEL_ID ?= qwen3-8b-q4-k-m
EMACSCLIENT ?= emacsclient

.PHONY: demo doctor probe smoke recursion classes classes-all frontier analyze export \
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
	@echo "Remote benchmark is never implicit. Run: ./scripts/bench --preset frontier --deployment remote --allow-paid --model-id MODEL_ID"

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
