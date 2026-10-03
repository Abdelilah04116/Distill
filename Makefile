.PHONY: help install lint format types test check run demo clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Installe le paquet en mode developpement
	pip install -e ".[dev,serve,graph]"

lint:  ## Ruff
	ruff check distill tests

format:  ## Formate le code
	ruff format distill tests
	ruff check --fix distill tests

types:  ## Mypy
	mypy distill

test:  ## Tests unitaires et d'integration (sans GPU ni reseau)
	pytest -q

check: lint types test  ## Tout verifier, comme la CI

demo:  ## Valide la tache de demonstration et affiche le plan
	distill validate --task examples/email-triage/task.yaml
	distill run --task examples/email-triage/task.yaml --dry-run

run:  ## Lance le pipeline complet sur la tache de demonstration
	distill run --task examples/email-triage/task.yaml

clean:  ## Supprime les artefacts de build et de test
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
