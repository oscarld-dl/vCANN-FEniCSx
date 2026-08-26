#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./fenicsx/paraview/open_in_paraview.sh [path/to/result.xdmf]
  ./fenicsx/paraview/open_in_paraview.sh --dry-run [path/to/result.xdmf]

If no .xdmf path is provided, the newest .xdmf file in ./fenicsx/results is used.

Optional environment variable:
  PARAVIEW_EXE
    Override the detected Linux ParaView executable path.
EOF
}

find_latest_xdmf() {
    local search_dir=$1
    find "$search_dir" -maxdepth 1 -type f -name '*.xdmf' -printf '%T@ %p\n' \
        | sort -nr \
        | head -n1 \
        | cut -d' ' -f2-
}

find_paraview_exe() {
    if [[ -n "${PARAVIEW_EXE:-}" ]]; then
        printf '%s\n' "$PARAVIEW_EXE"
        return 0
    fi

    command -v paraview
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
results_dir=$(cd -- "$script_dir/../results" && pwd)
xdmf_path=""
dry_run=0

while (($# > 0)); do
    case "$1" in
        --dry-run|--print)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$xdmf_path" ]]; then
                echo "Only one .xdmf path may be provided." >&2
                usage >&2
                exit 2
            fi
            xdmf_path=$1
            ;;
    esac
    shift
done

if [[ -z "$xdmf_path" ]]; then
    xdmf_path=$(find_latest_xdmf "$results_dir")
    if [[ -z "$xdmf_path" ]]; then
        echo "No .xdmf files found in $results_dir." >&2
        exit 1
    fi
fi

if [[ ! -f "$xdmf_path" ]]; then
    echo "File not found: $xdmf_path" >&2
    exit 1
fi

xdmf_path=$(realpath "$xdmf_path")
paraview_exe=$(find_paraview_exe || true)

if [[ -z "$paraview_exe" ]]; then
    echo "Could not find ParaView on this Linux system." >&2
    echo "Install ParaView or set PARAVIEW_EXE to the desired binary." >&2
    exit 1
fi

if [[ ! -x "$paraview_exe" ]]; then
    echo "ParaView executable is not runnable: $paraview_exe" >&2
    exit 1
fi

if ((dry_run)); then
    printf 'ParaView executable: %s\n' "$paraview_exe"
    printf 'Result file: %s\n' "$xdmf_path"
    exit 0
fi

nohup "$paraview_exe" "$xdmf_path" >/dev/null 2>&1 &
printf 'Opened %s in Linux ParaView.\n' "$xdmf_path"
