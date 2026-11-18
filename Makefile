.PHONY: setup test demo up down logs
setup:
	python -m pip install -e '.[test]'
test:
	pytest -q
demo:
	python scripts/run_local_stack.py --once
up:
	docker compose up --build -d --wait
down:
	docker compose down
logs:
	docker compose logs -f
