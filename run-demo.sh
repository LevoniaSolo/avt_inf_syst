#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python || ! -x .venv/bin/finautomation ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv_command=uv
    else
        uv_command=./.venv-tools/bin/uv
        if [[ ! -x "$uv_command" ]]; then
            printf '%s\n' 'Устанавливаю uv в локальное окружение .venv-tools.' >&2
            python3 -m venv .venv-tools
            .venv-tools/bin/python -m pip install uv==0.13.9
        fi
    fi
    "$uv_command" sync --locked
fi

exec .venv/bin/python -m finautomation.main run \
    --tickers examples/demo/tickers.csv \
    --financials examples/demo/financials.csv \
    --quotes examples/demo/quotes.csv \
    --config examples/demo/config.yaml \
    --synthetic-demo --out runs/demo
