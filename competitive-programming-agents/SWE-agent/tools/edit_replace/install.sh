bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH="$bundle_dir/lib":$PYTHONPATH

_write_env "CURRENT_FILE" "${CURRENT_FILE:-}"
_write_env "CURRENT_LINE" "${CURRENT_LINE:-0}"
_write_env "WINDOW" "$WINDOW"

pip install flake8
