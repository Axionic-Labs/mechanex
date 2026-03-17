.PHONY: lint type test test-integration coverage quality

lint:
	ruff check mechanex tests

type:
	mypy

test:
	python -m unittest discover -s tests -v

coverage:
	coverage run -m unittest discover -s tests -v
	coverage report --fail-under=70

test-integration:
	python -m unittest discover -s tests/integration -v

quality: lint type coverage
