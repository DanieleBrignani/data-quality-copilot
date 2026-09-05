#!/usr/bin/env bash
# Container entrypoint: wait for PostgreSQL, migrate, generate demo data, start the app.
#
# Migrations run here rather than in the image build because the database does not exist
# at build time. Alembic is idempotent, so a restarted container is a no-op.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }

wait_for_database() {
  local attempts=${DB_WAIT_ATTEMPTS:-30}
  local delay=${DB_WAIT_DELAY:-2}

  log "waiting for the database..."
  for ((i = 1; i <= attempts; i++)); do
    if python -c "
import sys
from dqcopilot.db.session import check_connection
sys.exit(0 if check_connection() else 1)
" 2>/dev/null; then
      log "database is ready"
      return 0
    fi
    sleep "${delay}"
  done

  log "database did not become ready after $((attempts * delay))s"
  return 1
}

run_migrations() {
  log "applying database migrations"
  alembic upgrade head
  log "migrations applied"
}

seed_demo_data() {
  if [[ ! -f /app/data/demo/ground_truth.json ]]; then
    log "generating synthetic demo datasets"
    python scripts/generate_demo_data.py --output /app/data/demo >/dev/null
  fi
}

main() {
  if [[ "${PERSISTENCE_ENABLED:-true}" == "true" ]]; then
    if wait_for_database; then
      run_migrations
    else
      # Persistence is optional by design: the app still profiles, corrects and
      # exports without it, so a missing database degrades rather than blocks.
      log "starting without persistence - the audit trail will not be written"
    fi
  else
    log "persistence disabled by configuration"
  fi

  seed_demo_data
  log "starting: $*"
  exec "$@"
}

main "$@"
