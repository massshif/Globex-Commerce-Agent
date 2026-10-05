.PHONY: check test backend-check frontend-check eval-offline

backend-check:
	.venv/bin/ruff check app scripts tests
	.venv/bin/python -m compileall -q app scripts
	.venv/bin/pytest -q

frontend-check:
	cd frontend && npm run build

eval-offline:
	.venv/bin/python -m scripts.eval.harness.run

test: backend-check

check: backend-check eval-offline frontend-check
