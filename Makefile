# LoadGuard development commands.

PYTHON ?= python

.PHONY: test coverage mypy lint format check demo benchmark pilot serve

test:  ## Run the unit tests
	$(PYTHON) -m unittest discover -s tests

coverage:  ## Run tests and report branch coverage (fails under 100%)
	$(PYTHON) -m coverage run --branch --source=src/loadguard -m unittest discover -s tests
	$(PYTHON) -m coverage report --fail-under=100

mypy:  ## Type-check the whole project
	mypy src mcp_server app.py

lint:  ## Lint and check formatting
	ruff check src tests demo scripts app.py mcp_server
	ruff format --check src tests demo scripts app.py mcp_server

format:  ## Apply ruff formatting
	ruff format src tests demo scripts app.py mcp_server

check: coverage mypy lint  ## Full gate (tests + coverage + types + lint)

demo:  ## Run the zero-dependency CLI demo
	$(PYTHON) demo/demo.py

benchmark:  ## Run the benchmark evaluation
	$(PYTHON) demo/benchmark.py

pilot:  ## Run the 3-phase pilot evaluation
	$(PYTHON) demo/benchmark.py --pilot demo/sample_events.jsonl

serve:  ## Launch the live interactive web dashboard
	$(PYTHON) app.py
