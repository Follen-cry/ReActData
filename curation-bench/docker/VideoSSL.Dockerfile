# Supply a pinned image already containing your InternVL-U environment and Codex.
# No download of model weights or credentials occurs at build time.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# Keep the uv bootstrap version aligned with the committed lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/
# This interpreter must be >=3.12. It can differ from profile.python (GPU).
ARG METHOD_PYTHON=/opt/conda/bin/python
WORKDIR /opt/video-ssl-method
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY src ./src
COPY vendor/video_ssl_backend ./vendor/video_ssl_backend
ENV UV_PROJECT_ENVIRONMENT=/opt/video-ssl-method/.venv
RUN uv sync --locked --no-python-downloads --python ${METHOD_PYTHON} \
    && command -v codex \
    && mkdir -p /tmp/video-ssl-home/.codex \
    && chmod 1777 /tmp/video-ssl-home /tmp/video-ssl-home/.codex
ENV PYTHONUNBUFFERED=1
ENV HOME=/tmp/video-ssl-home
