"""Detection and repair of text that was decoded with the wrong codec.

When a UTF-8 file is read as if it were cp1252, ``à`` (bytes ``C3 A0``) becomes the two
characters ``Ã`` + no-break space. The damage is silent: the file still parses, the
column still looks textual, and no other check notices. It is also *provable* rather
than guessable - the mangled text can be encoded back to the original bytes, and those
bytes are valid UTF-8. That round trip is what this module tests, which is why it does
not need a heuristic list of suspicious character pairs.

Two failure modes are distinguished, because only one of them can be undone:

* **Recoverable.** The original bytes survived; ``repair_mojibake`` returns the fixed
  text.
* **Lost.** The decoder already replaced the byte with U+FFFD. The information is gone
  and no correction can bring it back, so it is reported and never "fixed".
"""

from __future__ import annotations

#: Codecs a UTF-8 file is most often misread as. cp1252 is tried first because it is
#: the Windows default and therefore the usual culprit; latin-1 differs from it only in
#: the 0x80-0x9F range but accepts every byte, so it must come second or it would claim
#: every string.
_SUSPECT_CODECS: tuple[str, ...] = ("cp1252", "latin-1")

#: U+FFFD. Its presence means a decoder already gave up on a byte it could not map.
REPLACEMENT_CHARACTER = "�"

#: C1 control characters. Real text does not contain these, so a "repair" that produces
#: one has found a coincidence rather than a corruption.
_C1_RANGE = range(0x80, 0xA0)


def repair_mojibake(value: str) -> str | None:
    """Return ``value`` decoded correctly, or ``None`` when it is not mojibake.

    A value is treated as mojibake only when it can be encoded back to bytes with one
    of the suspect codecs *and* those bytes are valid UTF-8 that decodes to something
    different. Both conditions are strict: an accented word that was read correctly
    fails the second step, because its bytes are not a valid UTF-8 sequence.

    Args:
        value: The possibly damaged text.

    Returns:
        The repaired text, or ``None`` when there is nothing to repair.
    """
    if value.isascii() or not value:
        return None

    for codec in _SUSPECT_CODECS:
        try:
            candidate = value.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if candidate == value or not _is_plausible(candidate):
            continue
        return candidate
    return None


def has_lost_characters(value: str) -> bool:
    """True when the text contains U+FFFD, meaning a byte was already discarded."""
    return REPLACEMENT_CHARACTER in value


def _is_plausible(candidate: str) -> bool:
    """Reject a repair that produced control characters or fresh replacement marks."""
    if REPLACEMENT_CHARACTER in candidate:
        return False
    return not any(ord(character) in _C1_RANGE for character in candidate)
