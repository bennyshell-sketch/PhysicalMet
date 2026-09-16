# Dive - container image for hosted deployment.
#
# Deliberate choices:
#   * runs as a non-root user, so a compromise inside the app does not own the
#     container
#   * dependencies are installed in their own layer, so code changes rebuild in
#     seconds rather than minutes
#   * no data is baked in - cohort data arrives on a mounted volume, so the
#     image itself is never sensitive and can be pushed to a registry
#   * a healthcheck, so an orchestrator can tell a wedged container from a
#     healthy one

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is for the healthcheck; build-essential is dropped again after the wheels
# that need it are built.
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY dive/ ./dive/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY settings.yaml ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh

# Unprivileged runtime user. /data is the only writable path it needs.
RUN useradd --create-home --uid 10001 dive \
 && mkdir -p /data \
 && chown -R dive:dive /app /data
USER dive

ENV DIVE_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# The command line is assembled at start-up from the environment, because the
# public hostname decides the websocket Host allow-list. See
# docker-entrypoint.sh for what each flag is protecting against.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
