#!/bin/bash
# Open the drone depth web viewer in the host's default browser.
ONBOARD=${RS_HOST:-192.168.1.208}
URL="http://${ONBOARD}:8090"
echo "opening $URL"
xdg-open "$URL" >/dev/null 2>&1 || sensible-browser "$URL" >/dev/null 2>&1 || \
  echo "please open $URL manually"
