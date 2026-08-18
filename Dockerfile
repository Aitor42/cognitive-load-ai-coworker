FROM python:3.11-slim

WORKDIR /app

# Install LoadGuard with the optional API + LLM + MCP layers.
# pyproject.toml is the single source of truth for dependencies.
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Run as non-root for security.
RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "app.py"]
