#!/usr/bin/env bash
# Script to backup the TinyDB data file

set -e

BACKUP_DIR="data/backups"
DB_FILE="data/development.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

if [ -f "$DB_FILE" ]; then
    cp $DB_FILE "$BACKUP_DIR/backup_$TIMESTAMP.json"
    echo "Database backup created at $BACKUP_DIR/backup_$TIMESTAMP.json"
else
    echo "Database file $DB_FILE does not exist. Skipping backup."
fi
