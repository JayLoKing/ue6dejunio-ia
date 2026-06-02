#!/usr/bin/env bash
# Configura el entorno de entrenamiento dentro de WSL2 (Ubuntu).
# Ejecutar DENTRO de WSL, desde la carpeta del proyecto:
#   bash scripts/setup_wsl.sh
set -euo pipefail

# Entrenamiento en CPU: no se requiere GPU ni driver NVIDIA.

PYVER=3.11
# IMPORTANTE: el venv vive en el filesystem LINUX ($HOME), NO en /mnt/d.
# Crear venvs sobre /mnt/* (disco Windows) rompe ensurepip/pip en WSL.
VENV_DIR="${HOME}/.venv-ue6"

echo ">>> Instalando Python ${PYVER} (deadsnakes) si falta..."
if ! command -v python${PYVER} >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
fi
# Asegurar venv/dev aunque python3.11 ya exista (necesario para ensurepip).
sudo apt-get install -y python${PYVER} python${PYVER}-venv python${PYVER}-dev

echo ">>> Creando venv en ${VENV_DIR} (filesystem Linux)..."
rm -rf "${VENV_DIR}"
python${PYVER} -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ">>> Actualizando pip e instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> Verificando entorno..."
python scripts/check_env.py

echo ">>> Listo. Activa con:  source ${VENV_DIR}/bin/activate"
