#!/usr/bin/env bash
# Configura el entorno de entrenamiento en Linux nativo (probado para Linux Mint
# y Ubuntu). Ejecutar desde la carpeta del proyecto:
#
#   bash scripts/setup_linux.sh
#
# En Linux NO hace falta WSL: TF Decision Forests tiene wheel nativa. Es el mismo
# entorno que en WSL2, sin la capa del medio.
set -euo pipefail

PYVER=3.11
VENV_DIR="${HOME}/.venv-ue6"

# Linux Mint usa sus propios nombres de version (wilma, virginia, vanessa...) y
# deadsnakes solo publica para los de Ubuntu. Agregando el PPA sin traducir, apt
# queda apuntando a un repositorio que no existe y falla el update entero.
# /etc/os-release trae UBUNTU_CODENAME justamente para esto.
codename_ubuntu() {
  # shellcheck disable=SC1091
  . /etc/os-release
  if [ -n "${UBUNTU_CODENAME:-}" ]; then
    echo "${UBUNTU_CODENAME}"
  else
    echo "${VERSION_CODENAME:-}"
  fi
}

instalar_python() {
  if command -v "python${PYVER}" >/dev/null 2>&1; then
    echo ">>> Python ${PYVER} ya esta instalado."
  else
    local codename
    codename="$(codename_ubuntu)"
    if [ -z "${codename}" ]; then
      echo "!!! No se pudo determinar el codename de Ubuntu base." >&2
      echo "    Instala Python ${PYVER} a mano (pyenv o deadsnakes) y volve a correr." >&2
      exit 1
    fi
    echo ">>> Agregando deadsnakes para '${codename}' (base Ubuntu de esta distro)..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common ca-certificates
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu ${codename} main" \
      | sudo tee /etc/apt/sources.list.d/deadsnakes.list >/dev/null
    sudo apt-key adv --keyserver keyserver.ubuntu.com \
      --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 2>/dev/null \
      || sudo apt-get install -y gnupg
    sudo apt-get update
  fi

  # venv y dev aunque el binario ya exista: sin ellos falla ensurepip.
  sudo apt-get install -y "python${PYVER}" "python${PYVER}-venv" "python${PYVER}-dev"
}

instalar_python

echo ">>> Creando venv en ${VENV_DIR}..."
rm -rf "${VENV_DIR}"
"python${PYVER}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ">>> Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo ">>> Creando .env desde la plantilla. EDITALO: UE6_DATA_ROOT y UE6_API_TOKEN."
  cp .env.example .env
fi

echo ">>> Verificando entorno..."
python scripts/check_env.py || true

cat <<EOF

Listo. Para usarlo:

    source ${VENV_DIR}/bin/activate
    export PYTHONPATH=src
    python -m ue6_ia.cli all

Antes, edita .env:
  UE6_DATA_ROOT  -> la carpeta que contiene 'REGISTROS DE AÑOS PASADOS' y
                    'Registros 2026'. En Linux es una ruta normal, sin /mnt/:
                    por ejemplo /home/$(whoami)/ue6/DOCS docentes
  UE6_API_TOKEN  -> cualquier cadena larga; sin ella el servicio no atiende.
EOF
