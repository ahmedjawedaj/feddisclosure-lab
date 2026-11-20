# Flower Deployment Runtime, deliberately pinned to the application's Message API.
# No data/, evaluator truth, .git, or research artifacts are copied to this image.
FROM flwr/superexec:1.27.0
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
# CPU PyTorch. Install independently so pip does not pull CUDA wheels for the lab.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1' \
    && python -m pip install --no-cache-dir -e . 'scikit-learn>=1.4,<2'
COPY docker/flower/flwr-config.toml /opt/fdl-flwr-home/.flwr/config.toml
RUN chmod -R a+rwX /opt/fdl-flwr-home
ENV HOME=/opt/fdl-flwr-home \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
