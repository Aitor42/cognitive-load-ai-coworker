# LoadGuard development commands.

PYTHON ?= python

.PHONY: test coverage mypy lint format check

test:  ## Run the unit tests
	$(PYTHON) -m unittest discover -s tests

coverage:  ## Run tests and report branch coverage
	$(PYTHON) -m coverage run --branch --source=src/loadguard -m unittest discover -s tests
	$(PYTHON) -m coverage report

mypy:  ## Type-check the whole project
	mypy src mcp_server app.py

lint:  ## Lint and check formatting
	ruff check src tests demo scripts app.py mcp_server
	ruff format --check src tests demo scripts app.py mcp_server

format:  ## Apply ruff formatting
	ruff format src tests demo scripts app.py mcp_server

check: coverage mypy lint  ## Full gate (tests + coverage + types + lint)
