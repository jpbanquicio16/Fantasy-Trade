#!/bin/sh
# Nightly job (design section 8): refresh data, run the whole-league search, store the cards for the app.
# Schedule with scripts/com.trade.nightly.plist (launchd) or cron; a sleeping Mac skips runs.
set -e
cd "$(dirname "$0")/.."
.venv/bin/trade nightly >> data/nightly.log 2>&1
