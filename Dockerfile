# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/work/input/huggingface \
    HF_HUB_CACHE=/work/input/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TZ=Europe/Paris

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 nyx \
    && useradd --uid 10001 --gid nyx --create-home nyx \
    && mkdir -p /work/input /work/output /work/cache \
    && chown -R nyx:nyx /work

FROM base AS smoke
# Cheap image target for CLI/source checks without scientific installations.
COPY --chown=nyx:nyx production/ /app/production/
COPY --chown=nyx:nyx nyx_runtime/ /app/nyx_runtime/
USER 10001:10001
RUN mkdir -p /app/production/runtime/data /app/production/runtime/runs
ENTRYPOINT ["python", "-m", "nyx_runtime"]
CMD ["--help"]

FROM base AS runtime
COPY requirements.txt requirements_aws.txt ./
# proxy_ca is optional on a normal workstation; managed cloud builds should pass
# --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt (combined CA bundle).
# The CA and any private package requirements never become image layers.
RUN --mount=type=secret,id=proxy_ca \
    set -eu; \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --no-cache-dir 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements_aws.txt
RUN --mount=type=secret,id=proxy_ca \
    --mount=type=secret,id=private_requirements \
    --mount=type=secret,id=saturn_wheel,target=/run/secrets/tshistory_lite-0.5-py3-none-any.whl \
    set -eu; \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip freeze --all > /tmp/nyx-public-constraints.txt; \
    if [ -f /run/secrets/private_requirements ]; then \
      pip install --no-cache-dir --constraint /tmp/nyx-public-constraints.txt -r /run/secrets/private_requirements; \
    fi \
    && if [ -f /run/secrets/tshistory_lite-0.5-py3-none-any.whl ]; then \
      pip install --no-cache-dir --constraint /tmp/nyx-public-constraints.txt /run/secrets/tshistory_lite-0.5-py3-none-any.whl; \
    fi \
    && rm -f /tmp/nyx-public-constraints.txt

# The retained runtime writes resumable state inside its runtime directory.
# The wrapper restores/persists that state separately from immutable seed inputs.
COPY --from=smoke --chown=nyx:nyx /app/production/ /app/production/
COPY --from=smoke --chown=nyx:nyx /app/nyx_runtime/ /app/nyx_runtime/
USER 10001:10001
RUN python -c 'from nyx_runtime.runtime import verify_runtime; verify_runtime()'
ENTRYPOINT ["python", "-m", "nyx_runtime"]
CMD ["forecast", "--input-root", "/work/input", "--output-root", "/work/output"]
