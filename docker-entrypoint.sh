#!/bin/sh
# Builds Streamlit's command line from the deployment's environment.
#
# Two things have to be right for an internet-facing deployment, and both
# depend on the public hostname, which is only known at run time:
#
#   * server.allowedHosts - Streamlit refuses a websocket whose Host header is
#     not on this list, which is what stops someone pointing their own domain
#     at this container or reaching it directly by IP.
#   * browser.serverAddress - without it, headless Streamlit looks its own
#     external address up over the internet at every start. A deployment that
#     is meant to make no unsolicited outbound calls should not phone AWS to
#     print a banner nobody reads.
#
# Set DIVE_PUBLIC_HOSTNAME to the domain the reverse proxy serves, with no
# scheme and no port: dive.example.org
set -eu

HOSTNAME_PUBLIC="${DIVE_PUBLIC_HOSTNAME:-}"

set -- streamlit run app/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableXsrfProtection=true \
    --server.enableCORS=true \
    --browser.gatherUsageStats=false

if [ -n "$HOSTNAME_PUBLIC" ]; then
    set -- "$@" \
        --browser.serverAddress="$HOSTNAME_PUBLIC" \
        --server.allowedHosts="$HOSTNAME_PUBLIC"
else
    # No public hostname given: assume a local or port-forwarded deployment.
    # Still pinned to localhost so the external-address lookup never happens.
    echo "dive: DIVE_PUBLIC_HOSTNAME is not set - the Host allow-list is off." >&2
    set -- "$@" --browser.serverAddress=localhost
fi

exec "$@"
