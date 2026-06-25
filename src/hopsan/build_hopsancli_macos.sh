#!/usr/bin/env bash
# Build HopsanCLI (head-less, no Qt/GUI) from source on macOS (Apple Silicon).
#
# There is no official macOS Hopsan package, but HopsanCore + HopsanCLI are
# plain C++ and build cleanly with the Xcode command-line tools + CMake. This
# script reproduces the local install used by cylinder_sim/hopsan_plant.py.
#
# Requirements: git, cmake, a C/C++ compiler (Xcode CLT). gmake/automake are
# pulled in by Homebrew for the tclap setup step.
#
# Result: ~/src/hopsan-build/HopsanCLI/hopsancli  (symlinked onto PATH)
#         + libdefaultcomponentlibrary.dylib (the component library, loaded -e)
set -euo pipefail

SRC="${HOME}/src/hopsan"
BUILD="${HOME}/src/hopsan-build"
JOBS="${JOBS:-8}"

echo "==> 1/5 clone (shallow + submodules)"
mkdir -p "${HOME}/src"
if [ ! -d "${SRC}/.git" ]; then
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/Hopsan/hopsan.git "${SRC}"
fi

echo "==> 2/5 fetch tclap (header-only dep needed by the CLI)"
( cd "${SRC}/dependencies" && ./setupTclap.sh )

echo "==> 3/5 trim top-level CMake to CLI-only targets (skip Qt/GUI)"
python3 - "${SRC}/CMakeLists.txt" << 'PY'
import sys
f = sys.argv[1]
keep = {"HopsanCore", "componentLibraries", "HopsanCLI"}
out = []
for line in open(f):
    s = line.strip()
    if s.startswith("add_subdirectory("):
        name = s[s.find("(")+1:s.find(")")].strip()
        if name not in keep:
            out.append("# [CLI-build disabled] " + line.rstrip("\n") + "\n")
            continue
    out.append(line)
open(f, "w").writelines(out)
print("   kept:", sorted(keep))
PY

echo "==> 4/5 configure + build"
rm -rf "${BUILD}"; mkdir -p "${BUILD}"
# CMAKE_POLICY_VERSION_MINIMUM=3.5 lets CMake 4.x accept Hopsan's old minimums.
cmake -S "${SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "${BUILD}" --config Release --parallel "${JOBS}"

echo "==> 5/5 put hopsancli on PATH"
LINKDIR="$(brew --prefix 2>/dev/null)/bin"; LINKDIR="${LINKDIR:-/usr/local/bin}"
ln -sf "${BUILD}/HopsanCLI/hopsancli" "${LINKDIR}/hopsancli"

echo
echo "Done. hopsancli -> ${BUILD}/HopsanCLI/hopsancli  (linked in ${LINKDIR})"
"${BUILD}/HopsanCLI/hopsancli" --version || true
echo "Component library: ${BUILD}/componentLibraries/defaultLibrary/libdefaultcomponentlibrary.dylib"
echo "The Python driver auto-discovers both (no env vars needed)."
