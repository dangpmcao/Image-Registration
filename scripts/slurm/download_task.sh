#!/usr/bin/env bash
set -euo pipefail
root="$1"
task="$2"
data="$root/dataset"
case "$task" in
    0) folder=learn2reg_lungct; archive=learn2reg_lungct.zip; url=https://cloud.imi.uni-luebeck.de/s/o7LyCbJCie8fQ3B/download ;;
    1) folder=learn2reg_oasis; archive=OASIS.zip; url=https://cloud.imi.uni-luebeck.de/s/xcZrLSQYtK68em8/download ;;
    2) folder=learn2reg_abdomenctct; archive=AbdomenCTCT.zip; url=https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT/download ;;
    3) folder=learn2reg_nlst2023; archive=NLST2023.zip; url=https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR/download ;;
    4|5)
        modality=T1
        if [[ "$task" == 5 ]]; then modality=T2; fi
        folder="ixi/$modality"
        archive="IXI-${modality}.tar"
        url="https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/$archive"
        ;;
    6) folder=tcia_breast_nact ;;
    *) echo "Invalid task: $task" >&2; exit 2 ;;
esac
mkdir -p "$data/$folder"
exec 9>"$data/$folder/.download.lock"
flock -n 9 || { echo "Another writer is active for $folder" >&2; exit 1; }
cd "$data/$folder"
echo "START $(date -Is) host=$(hostname) dataset=$folder job=${SLURM_JOB_ID:-local}"
if [[ -f .download_complete ]]; then
    echo "Already completed: $(cat .download_complete)"
    exit 0
fi
if [[ "$task" == 6 ]]; then
    env_dir="$data/.download_idc_env"
    if [[ ! -x "$env_dir/bin/python" ]]; then
        /home/dang.cpm/miniconda3/bin/python -m venv "$env_dir"
    fi
    "$env_dir/bin/python" -m pip install --disable-pip-version-check idc-index pandas
    "$env_dir/bin/python" -u "$root/Image-Registration/scripts/slurm/download_breast.py" "$data/$folder"
    "$env_dir/bin/python" -m pip freeze > .download_packages.txt
else
    valid_archive() {
        if [[ "$archive" == *.zip ]]; then
            unzip -tq "$archive" >/dev/null 2>&1
        else
            tar -tf "$archive" >/dev/null 2>&1
        fi
    }
    if ! valid_archive; then
        tls_options=()
        if [[ "$task" == 4 || "$task" == 5 ]]; then
            # The official IXI host's certificate is expired (checked 2026-11-28).
            # Exception applies only to these public archives; no credentials.
            echo "IXI: using official HTTPS host with expired-certificate exception; archive authenticity is not verified."
            tls_options=(--insecure)
        fi
        curl --fail --location --continue-at - --retry 10 --retry-delay 5 \
            --connect-timeout 30 --speed-limit 1024 --speed-time 300 \
            "${tls_options[@]}" --output "$archive" "$url"
    fi
    # Validate before extraction; retain archives for recovery.
    valid_archive
    if [[ "$archive" == *.zip ]]; then
        unzip -oq "$archive"
    else
        tar -xf "$archive"
    fi
fi
date -Is > .download_complete
echo "COMPLETE $(date -Is) dataset=$folder"
