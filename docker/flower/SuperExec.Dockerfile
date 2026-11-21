# Flower Deployment Runtime, deliberately pinned to the application's Message API.
# No data/, evaluator truth, .git, or research artifacts are copied to this image.
FROM flwr/superexec:1.27.0
# The base image runs as the unprivileged 'app' user. /app must belong to it so the
# editable install can write its metadata; nothing here ever runs as root at runtime.
USER root
RUN mkdir -p /app /opt/fdl-flwr-home/.flwr && chown -R app:app /app /opt/fdl-flwr-home
USER app
WORKDIR /app
COPY --chown=app:app --chmod=a+rX pyproject.toml README.md ./
COPY --chown=app:app --chmod=a+rX src/ ./src/
# CPU PyTorch. Install independently so pip does not pull CUDA wheels for the lab.
# Optional BuildKit secret "fdl_build_ca": a PEM bundle for TLS-inspecting egress
# proxies. It exists only during this RUN step and is never stored in an image
# layer. An empty file (the default placeholder) means "use the image trust store".
RUN --mount=type=secret,id=fdl_build_ca,mode=0444 \
    if [ -s /run/secrets/fdl_build_ca ]; then export PIP_CERT=/run/secrets/fdl_build_ca; fi \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1' \
    && python -m pip install --no-cache-dir -e . 'scikit-learn>=1.4,<2'
COPY --chown=app:app --chmod=a+rX docker/flower/flwr-config.toml /opt/fdl-flwr-home/.flwr/config.toml
RUN chmod -R a+rwX /opt/fdl-flwr-home
ENV HOME=/opt/fdl-flwr-home \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
