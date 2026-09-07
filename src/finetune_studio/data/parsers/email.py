"""Email (.eml / .msg) parser — RFC 822 headers + body."""

from __future__ import annotations

import email
import email.policy
from email import message_from_string
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        msg = message_from_string(raw, policy=email.policy.default)
    except Exception as e:  # noqa: BLE001
        return make_result(raw, {"type": "email", "error": str(e)},
                           parser="email_v1", warnings=[f"email parse failed: {e}"])
    headers = {k: str(v) for k, v in msg.items()}
    parts_text = []
    attachments = []
    for part in msg.walk():
        ctype = part.get_content_type()
        disp = str(part.get("Content-Disposition") or "")
        if part.is_multipart():
            continue
        if "attachment" in disp:
            attachments.append({
                "filename": part.get_filename(),
                "content_type": ctype,
                "size": len(part.get_payload(decode=True) or b""),
            })
            continue
        try:
            body = part.get_content()
        except Exception:
            body = part.get_payload()
        if isinstance(body, str) and body.strip():
            parts_text.append(f"[{ctype}]\n{body}")
    text = "\n\n".join(parts_text) if parts_text else msg.get_body(preferencelist=("plain",))
    if not text:
        text = raw  # last resort
    structured = {
        "type": "email",
        "subject": msg.get("Subject", ""),
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "cc": msg.get("Cc", ""),
        "date": msg.get("Date", ""),
        "message_id": msg.get("Message-ID", ""),
        "headers": headers,
        "attachment_count": len(attachments),
        "attachments": attachments,
    }
    return make_result(text, structured, parser="email_v1")


if __name__ == "__main__":
    cli_run(parse)
