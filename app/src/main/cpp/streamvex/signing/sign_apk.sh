#!/usr/bin/env bash
# ==============================================================================
# StreamVex Master APK Signing Script
# Uses the permanent master keystore streamvex_master.jks to sign Solus APKs.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEYSTORE="${SCRIPT_DIR}/streamvex_master.jks"
KEY_ALIAS="streamvex_key"
KEY_PASS="streamvex2026"
STORE_PASS="streamvex2026"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input-apk> [output-apk]"
    echo "Example: $0 app-release-unsigned.apk Solus-Release-Signed.apk"
    exit 1
fi

INPUT_APK="$1"
OUTPUT_APK="${2:-${INPUT_APK%.apk}-signed.apk}"

if [[ ! -f "${INPUT_APK}" ]]; then
    echo "[-] Error: Input APK not found: ${INPUT_APK}"
    exit 1
fi

if [[ ! -f "${KEYSTORE}" ]]; then
    echo "[-] Error: Master keystore not found: ${KEYSTORE}"
    exit 1
fi

echo "[+] Signing APK: ${INPUT_APK}"
echo "[+] Keystore:   ${KEYSTORE} (alias: ${KEY_ALIAS})"
echo "[+] Output:     ${OUTPUT_APK}"

apksigner sign \
    --v1-signing-enabled true \
    --v2-signing-enabled true \
    --v3-signing-enabled true \
    --ks "${KEYSTORE}" \
    --ks-key-alias "${KEY_ALIAS}" \
    --ks-pass "pass:${STORE_PASS}" \
    --key-pass "pass:${KEY_PASS}" \
    --out "${OUTPUT_APK}" \
    "${INPUT_APK}"

echo "[+] Verifying signature..."
apksigner verify --verbose "${OUTPUT_APK}"

echo "[+] SUCCESS: APK signed successfully at: ${OUTPUT_APK}"
