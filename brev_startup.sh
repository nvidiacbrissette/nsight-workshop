#!/usr/bin/env bash
set -euo pipefail

# Brev startup script for the profiling workshop.
#
# This script is intended to run before Jupyter notebooks are exposed. It keeps
# setup in a project venv, registers that venv as a Jupyter kernel, and makes
# NVIDIA profiler binaries discoverable for notebook subprocess calls.

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV_DIR="${VENV_DIR:-"${SCRIPT_DIR}/.venv"}"
KERNEL_NAME="${KERNEL_NAME:-profiling-workshop}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (profiling-workshop)}"

log() {
  printf '\n[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
elif have sudo; then
  SUDO=(sudo)
else
  SUDO=()
fi

need_root() {
  if [ "$(id -u)" -ne 0 ] && ! have sudo; then
    die "This setup needs root privileges for apt/CUDA installation, but sudo is not available."
  fi
}

apt_get_update() {
  need_root
  "${SUDO[@]}" apt-get update
}

apt_install() {
  need_root
  "${SUDO[@]}" apt-get install -y --no-install-recommends "$@"
}

apt_has_candidate() {
  apt-cache policy "$1" 2>/dev/null | grep -q 'Candidate: [^()]'
}

detect_nvidia_repo_arch() {
  case "$(dpkg --print-architecture)" in
    amd64) printf 'x86_64' ;;
    arm64) printf 'sbsa' ;;
    *) return 1 ;;
  esac
}

detect_nvidia_repo_distro() {
  [ -r /etc/os-release ] || return 1
  # shellcheck disable=SC1091
  . /etc/os-release

  case "${ID:-}" in
    ubuntu)
      printf 'ubuntu%s' "$(printf '%s' "${VERSION_ID:-}" | tr -d '.')"
      ;;
    debian)
      printf 'debian%s' "${VERSION_ID%%.*}"
      ;;
    *)
      return 1
      ;;
  esac
}

install_nvidia_cuda_repo() {
  local distro arch url tmp_deb

  if ! have apt-get; then
    die "Automatic CUDA toolkit installation currently supports Debian/Ubuntu apt-based images."
  fi

  distro="$(detect_nvidia_repo_distro)" || die "Unsupported OS for automatic NVIDIA CUDA apt repo setup."
  arch="$(detect_nvidia_repo_arch)" || die "Unsupported architecture for automatic NVIDIA CUDA apt repo setup."
  url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb"
  tmp_deb="$(mktemp -p /tmp cuda-keyring.XXXXXX.deb)"

  log "Adding NVIDIA CUDA apt repository for ${distro}/${arch}"
  curl -fsSL "$url" -o "$tmp_deb"
  need_root
  "${SUDO[@]}" dpkg -i "$tmp_deb"
  rm -f "$tmp_deb"
  apt_get_update
}

first_matching_apt_package() {
  local pattern="$1"
  apt-cache search --names-only "$pattern" 2>/dev/null | awk '{print $1}' | sort -V | tail -n 1
}

find_tool_binary() {
  local tool="$1"

  if have "$tool"; then
    command -v "$tool"
    return 0
  fi

  find /opt/nvidia /usr/local/cuda /usr/local/cuda-* \
    -type f -name "$tool" -perm /111 2>/dev/null | sort -V | tail -n 1
}

link_tool_if_needed() {
  local tool="$1"
  local target

  if have "$tool"; then
    return 0
  fi

  target="$(find_tool_binary "$tool" || true)"
  if [ -z "$target" ]; then
    return 1
  fi

  need_root
  "${SUDO[@]}" ln -sf "$target" "/usr/local/bin/${tool}"
}

write_cuda_profile_path() {
  local tmp_profile

  if [ "$(id -u)" -ne 0 ] && ! have sudo; then
    warn "Skipping /etc/profile.d CUDA PATH setup because sudo is unavailable."
    return 0
  fi

  tmp_profile="$(mktemp -p /tmp profiling-workshop-cuda.XXXXXX.sh)"
  cat >"$tmp_profile" <<'EOF'
# Added by profiling-workshop/brev_startup.sh.
_profiling_workshop_prepend_path() {
  [ -d "$1" ] || return 0
  case ":${PATH:-}:" in
    *":$1:"*) ;;
    *) PATH="$1${PATH:+:${PATH}}" ;;
  esac
}

_profiling_workshop_prepend_ld_path() {
  [ -d "$1" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$1:"*) ;;
    *) LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

for _profiling_workshop_dir in \
  /usr/local/cuda/bin \
  /usr/local/cuda-*/bin \
  /opt/nvidia/nsight-systems/*/target-linux-x64 \
  /opt/nvidia/nsight-compute/*
do
  _profiling_workshop_prepend_path "$_profiling_workshop_dir"
done

for _profiling_workshop_dir in \
  /usr/local/cuda/lib64 \
  /usr/local/cuda-*/lib64
do
  _profiling_workshop_prepend_ld_path "$_profiling_workshop_dir"
done

export PATH
export LD_LIBRARY_PATH
unset -f _profiling_workshop_prepend_path
unset -f _profiling_workshop_prepend_ld_path
unset _profiling_workshop_dir
EOF

  need_root
  "${SUDO[@]}" install -m 0644 "$tmp_profile" /etc/profile.d/profiling-workshop-cuda.sh
  rm -f "$tmp_profile"
}

fix_workspace_ownership() {
  local owner

  if [ "$(id -u)" -ne 0 ]; then
    return 0
  fi

  owner="$(stat -c '%u:%g' "$SCRIPT_DIR")"
  chown -R "$owner" "$VENV_DIR" "${SCRIPT_DIR}/traces"
}

install_system_packages() {
  if ! have apt-get; then
    die "This startup script expects a Debian/Ubuntu Brev image with apt-get."
  fi

  log "Installing base OS packages"
  apt_get_update
  apt_install \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    pciutils \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

  if ! apt_has_candidate cuda-toolkit; then
    install_nvidia_cuda_repo
  fi

  log "Installing CUDA toolkit and NVIDIA profiling tools"
  apt_install cuda-toolkit

  if ! link_tool_if_needed nsys; then
    local nsys_pkg
    nsys_pkg="$(first_matching_apt_package '^(cuda-)?nsight-systems')"
    if [ -n "$nsys_pkg" ]; then
      apt_install "$nsys_pkg"
      link_tool_if_needed nsys || true
    fi
  fi

  if ! link_tool_if_needed ncu; then
    local ncu_pkg
    ncu_pkg="$(first_matching_apt_package '^(cuda-)?nsight-compute')"
    if [ -n "$ncu_pkg" ]; then
      apt_install "$ncu_pkg"
      link_tool_if_needed ncu || true
    fi
  fi

  link_tool_if_needed ncu-ui || true
  link_tool_if_needed nsys-ui || true
  write_cuda_profile_path
}

detect_driver_cuda_version() {
  if ! have nvidia-smi; then
    return 1
  fi

  nvidia-smi 2>/dev/null |
    sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' |
    head -n 1
}

detect_pytorch_index_url() {
  local version numeric

  if [ -n "${PYTORCH_INDEX_URL:-}" ]; then
    printf '%s' "$PYTORCH_INDEX_URL"
    return 0
  fi

  version="$(detect_driver_cuda_version || true)"
  if [ -z "$version" ]; then
    return 1
  fi

  numeric="$(awk -F. '{ print ($1 * 100) + $2 }' <<<"$version")"
  if [ "$numeric" -ge 1208 ]; then
    printf 'https://download.pytorch.org/whl/cu128'
  elif [ "$numeric" -ge 1206 ]; then
    printf 'https://download.pytorch.org/whl/cu126'
  elif [ "$numeric" -ge 1108 ]; then
    printf 'https://download.pytorch.org/whl/cu118'
  else
    return 1
  fi
}

gpu_expected() {
  have nvidia-smi || [ -e /dev/nvidiactl ] || [ -e /dev/nvidia0 ]
}

install_python_packages() {
  local pytorch_index
  local kernel_install_args

  log "Creating Python virtual environment at ${VENV_DIR}"
  python3 -m venv "$VENV_DIR"

  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"

  log "Installing Python packaging tools"
  python -m pip install --upgrade pip setuptools wheel

  pytorch_index="$(detect_pytorch_index_url || true)"
  if [ -n "$pytorch_index" ]; then
    log "Installing CUDA-capable PyTorch from ${pytorch_index}"
    python -m pip install --upgrade torch --index-url "$pytorch_index"
  else
    warn "Could not detect a CUDA driver version; installing PyTorch from the default package index."
    python -m pip install --upgrade torch
  fi

  log "Installing workshop Python requirements"
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

  log "Registering Jupyter kernel ${KERNEL_NAME}"
  kernel_install_args=(--name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME")
  if [ "$(id -u)" -eq 0 ]; then
    kernel_install_args+=(--prefix /usr/local)
  else
    kernel_install_args+=(--user)
  fi
  python -m ipykernel install "${kernel_install_args[@]}"
}

verify_setup() {
  local missing=0

  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"

  log "Verifying PyTorch CUDA availability"
  python - <<'PY'
import sys
import torch

print(f"torch: {torch.__version__}")
print(f"torch CUDA runtime: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

  if gpu_expected; then
    python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("CUDA-capable GPU was detected, but torch.cuda.is_available() is false.")
PY
  fi

  for tool in nsys ncu; do
    if ! have "$tool"; then
      warn "${tool} is not available on PATH."
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    die "Nsight Systems and Nsight Compute are required for this workshop."
  fi

  log "Nsight Systems"
  nsys --version

  log "Nsight Compute"
  ncu --version
}

main() {
  cd "$SCRIPT_DIR"
  mkdir -p traces

  install_system_packages
  install_python_packages
  fix_workspace_ownership
  verify_setup

  log "Brev startup setup complete"
  printf 'Use the Jupyter kernel named "%s" for the workshop notebooks.\n' "$KERNEL_DISPLAY_NAME"
}

main "$@"
