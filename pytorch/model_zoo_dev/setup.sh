#!/bin/bash

set -eo pipefail

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_CYAN='\033[1;36m'
  echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}

if [ -z ${SCRIPT_DIR+x} ]; then
  SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
else
  AML_DIR="${SCRIPT_DIR}/ampere_model_library"
fi

if [ ! -f "$SCRIPT_DIR/ampere_model_library/setup_deb.sh" ]; then
   log "Please pull submodules first: git submodule update --init --recursive"
   exit 1
fi

log "Checking for Debian or RHEL based Linux ..."
sleep 1
if [ -f "/etc/debian_version" ]; then
  debian_version=$(</etc/debian_version)
  log "Detected Debian $debian_version. Be advised that this script supports Debian >=11.0."
  sleep 3

  if [ -z ${AML_DIR+x} ]; then
    FORCE_INSTALL=1 bash $SCRIPT_DIR/ampere_model_library/setup_deb.sh
  else
    FORCE_INSTALL=1 SCRIPT_DIR=$AML_DIR bash $AML_DIR/setup_deb.sh
  fi

  log "Installing system dependencies ..."
  sleep 1
  apt-get update -y
  apt-get install -y zip curl vim numactl
  log "done.\n"
elif [ -f "/etc/redhat-release" ]; then
  rhel_version=$(</etc/redhat-release)
  log "Detected $rhel_version. Be advised that this script supports RHEL>=9.4."
  sleep 3

  if [ -z ${AML_DIR+x} ]; then
    FORCE_INSTALL=1 bash $SCRIPT_DIR/ampere_model_library/setup_rhel.sh
  else
    FORCE_INSTALL=1 SCRIPT_DIR=$AML_DIR bash $AML_DIR/setup_rhel.sh
  fi

  log "Installing system dependencies ..."
  sleep 1

  yum install -y zip curl vim numactl
  log "done.\n"
else
   log "\nNeither Debian-based nor RHEL-based Linux has been detected! Quitting."
   exit 1
fi

log "Installing python dependencies ..."
sleep 1
pip3 install --no-deps --upgrade python-Levenshtein==0.12.2
pip3 install O365==2.0.25
log "done.\n"

log "Authorizing HuggingFace Hub ..."
sleep 1
mkdir -p ~/.cache/huggingface
echo "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" > ~/.cache/huggingface/token
log "done.\n"

log "Setup completed. Please run: source $SCRIPT_DIR/ampere_model_library/set_env_variables.sh"
