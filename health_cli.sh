#!/bin/sh
set -eu
program_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -eq 0 ] || [ "$1" = "--json" ]; then
    exec /usr/bin/python3 "$program_dir/monitor.py" --status "$@"
fi
exec /usr/bin/python3 "$program_dir/monitor.py" "$@"
