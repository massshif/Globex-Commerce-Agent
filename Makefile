.PHONY: check test backend-check frontend-check eval-offline seed-demo

backend-check:
	.venv/bin/ruff check app scripts tests
	.venv/bin/python -m compileall -q app scripts
	.venv/bin/pytest -q

frontend-check:
	cd frontend && npm run build

eval-offline:
	.venv/bin/python -m scripts.eval.harness.run

test: backend-check

seed-demo:
	.venv/bin/python scripts/seed_demo_data.py

check: backend-check eval-offline frontend-check
