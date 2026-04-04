#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${HOME}"
PROJECT_REPO_URL="https://github.com/dmd0822/ml-pidog-gate_tuning.git"
PROJECT_DIR="${HOME_DIR}/ml-pidog-gate_tuning"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  local clone_args="${3:-}"

  if [ -d "${target_dir}/.git" ]; then
    echo "Updating existing ${target_dir}"
    (cd "${target_dir}" && git pull --ff-only)
    return 0
  fi

  echo "Cloning ${repo_url} -> ${target_dir}"
  # shellcheck disable=SC2086
  git clone ${clone_args} "${repo_url}" "${target_dir}"
}

sudo apt update
sudo apt install -y git python3-pip python3-setuptools python3-smbus

clone_if_missing "https://github.com/sunfounder/robot-hat.git" "${HOME_DIR}/robot-hat" "-b 2.5.x --depth=1"
(cd "${HOME_DIR}/robot-hat" && sudo python3 install.py)

clone_if_missing "https://github.com/sunfounder/vilib.git" "${HOME_DIR}/vilib" "--depth=1"
(cd "${HOME_DIR}/vilib" && sudo python3 install.py)

clone_if_missing "https://github.com/sunfounder/pidog.git" "${HOME_DIR}/pidog" "--depth=1"
(cd "${HOME_DIR}/pidog" && sudo rm -rf pidog.egg-info build dist)
python3 -m pip install "${HOME_DIR}/pidog" --no-build-isolation --no-deps --break-system-packages

if [ ! -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "Project not found at ${REPO_DIR}. Cloning into ${PROJECT_DIR}"
  clone_if_missing "${PROJECT_REPO_URL}" "${PROJECT_DIR}" ""
  REPO_DIR="${PROJECT_DIR}"
fi

python3 -m pip install -U pip --break-system-packages
python3 -m pip install "${REPO_DIR}" --break-system-packages

echo "Pi setup complete. Project installed from ${REPO_DIR}."
