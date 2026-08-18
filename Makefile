MODEL_ID ?= qwen35-4b

.PHONY: demo doctor probe smoke recursion classes frontier analyze export test clean clean-demo

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
	./scripts/bench --preset classes --deployment local --quiet
	./scripts/analyze

frontier:
	@echo "Remote benchmark is never implicit. Run: ./scripts/bench --preset frontier --deployment remote --allow-paid --model-id MODEL_ID"

analyze:
	./scripts/analyze

export:
	./scripts/export

test:
	python3 -m unittest discover -s tests -v

clean-demo:
	rm -f results/raw/demo-synthetic.jsonl

clean:
	rm -f results/raw/*.jsonl results/raw/*.manifest.json results/summary/*.csv results/summary/*.json results/plots/*.svg public/*.html
	printf '* Results dashboard\n\nNo analyzed benchmark data yet.\n' > results/summary/metrics.org
	printf '* Graphs\n\nNo plots generated yet.\n' > results/summary/plots.org
