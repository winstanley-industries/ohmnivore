"""Exact, local-only syntax adapter for the frozen proprietary SiC model.

The original and adapted model bytes are fetched/generated locally. Neither is
redistributed as part of the EMI-01 study or its retained evidence.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import zipfile


ARCHIVE_SHA256 = "de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51"
ARCHIVE_SIZE = 28318
MEMBER = "1200V-SMA-SiC-MOSFET-SPICE-Models/MSCSMA120.lib"
MEMBER_SHA256 = "6e888e103977f539e62b64952797ded391d2a49c59738bfd53f5fbe4b1fc8df7"
MEMBER_SIZE = 29805
ADAPTED_SHA256 = "17732ddd7ab5361073f8f23594e32158270af47d9b7188cf0ce773189cafcf96"
TEMP_TOKENS = 109
ADAPTER_VERSION = "emi01-microchip-ngspice-v1"


class ModelError(ValueError):
    """A model input cannot enter a qualified study."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _translate_text(text: str, expected_temp_tokens: int = TEMP_TOKENS) -> str:
    """Mechanical transformations; separate to test with independently written text."""
    # ngspice's PSpice compatibility reader reserves TEMP as ambient temperature.
    # This library instead defines and passes a fixed junction-temperature parameter.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, count = re.subn(r"\bTEMP\b", "TJ_C", text, flags=re.IGNORECASE)
    if count != expected_temp_tokens:
        raise ModelError(
            "unsupported_input", "unexpected local temperature token count"
        )
    for old, new, positive, negative in (
        ("Fgd", "Bgd", "42", "23"),
        ("Fds", "Bds", "42", "44"),
    ):
        # Keep spacing, output terminals and the whole expression after '=' intact.
        pattern = rf"^{old}(\s+{positive}\s+{negative}\s+)VALUE\s*="
        text, count = re.subn(
            pattern, rf"{new}\1I=", text, flags=re.IGNORECASE | re.MULTILINE
        )
        if count != 1:
            raise ModelError(
                "unsupported_input",
                f"expected exactly one {old} behavioral current source",
            )
    if re.search(r"^F\S*\s", text, flags=re.IGNORECASE | re.MULTILINE):
        raise ModelError("unsupported_input", "unadapted current-controlled source")
    return text


def adapt_bytes(member_bytes: bytes) -> bytes:
    """Adapt only the exact qualified library, including its original notices."""
    if len(member_bytes) != MEMBER_SIZE or _sha256(member_bytes) != MEMBER_SHA256:
        raise ModelError(
            "provenance_mismatch", "SiC model member does not match frozen bytes"
        )
    adapted = _translate_text(member_bytes.decode("cp1252")).encode("utf-8")
    if _sha256(adapted) != ADAPTED_SHA256:
        raise ModelError(
            "provenance_mismatch", "adapted model does not match frozen bytes"
        )
    return adapted


def adapt_archive(
    archive_path: str | Path, output_path: str | Path
) -> dict[str, str | int]:
    """Check the fetched archive and emit the adapted model in local scratch space.

    Hashes are suitable for retained provenance; the output file itself is not.
    No archive entries are extracted to the filesystem.
    """
    archive_path = Path(archive_path)
    try:
        if archive_path.stat().st_size != ARCHIVE_SIZE:
            raise ModelError(
                "provenance_mismatch",
                "SiC model archive size differs from frozen input",
            )
        archive = archive_path.read_bytes()
    except OSError as error:
        raise ModelError(
            "provenance_mismatch", "SiC model archive is unavailable"
        ) from error
    if _sha256(archive) != ARCHIVE_SHA256:
        raise ModelError(
            "provenance_mismatch", "SiC model archive does not match frozen bytes"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            matches = [info for info in package.infolist() if info.filename == MEMBER]
            if len(matches) != 1 or matches[0].file_size != MEMBER_SIZE:
                raise ModelError(
                    "provenance_mismatch",
                    "missing, duplicate or oversized model member",
                )
            member = package.read(matches[0])
    except (zipfile.BadZipFile, KeyError, RuntimeError) as error:
        raise ModelError("provenance_mismatch", "invalid SiC model archive") from error
    adapted = adapt_bytes(member)
    Path(output_path).write_bytes(adapted)
    return {
        "adapter_version": ADAPTER_VERSION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_SIZE,
        "member": MEMBER,
        "member_sha256": MEMBER_SHA256,
        "member_bytes": MEMBER_SIZE,
        "adapted_sha256": ADAPTED_SHA256,
        "adapted_bytes": len(adapted),
        "model_build_date": "2026-05-05",
        "model_internal_version": "2026.5",
    }
