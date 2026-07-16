FROM python:3.12-slim

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e ".[dev]"

# Default: run the multi-step demo.
CMD ["python", "-m", "agentgraph.cli", "demo"]
