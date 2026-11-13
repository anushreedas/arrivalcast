# CPU image for reproducing the experiments / serving forecasts.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[data]"

COPY data/synthetic ./data/synthetic
COPY data/bts/DL_JFK-LAX.parquet ./data/bts/DL_JFK-LAX.parquet

RUN useradd --create-home app && chown -R app /app
USER app
ENTRYPOINT ["arrivalcast"]
CMD ["--help"]
