#!/bin/sh
set -eu
program_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /usr/bin/python3 "$program_dir/safe_run.py" "$@"
