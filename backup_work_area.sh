#!/bin/bash
set -exuo pipefail

CUR_DATE="$(date '+%Y-%m-%dT%T' | tr -d ':')"
cd ./workarea

mkdir -p ../workarea_backups/

BACKUP_NAME="../workarea_backups/workarea_backup_$CUR_DATE.tar.zst"

tar cv . | zstd -z --ultra -T0 -22 > $BACKUP_NAME
