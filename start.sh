#!/bin/bash
# Start the Moral Dilemma Assessor backend server
# Usage: ./start.sh

cd "$(dirname "$0")"
PORT="${PORT:-8011}"
uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --reload
