#!/usr/bin/env python3
"""Send an alert email via Gmail SMTP.

Usage: send_alert_email.py "subject" "body"
Reads the app password from ~/.aston_smtp_password.
"""
import sys
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

FROM_ADDR = "nikhil.richard84@gmail.com"
TO_ADDR = "nikhil.richard84@gmail.com"
PASSWORD_FILE = Path.home() / ".aston_smtp_password"


def main():
    if len(sys.argv) != 3:
        print("usage: send_alert_email.py <subject> <body>", file=sys.stderr)
        sys.exit(2)
    subject, body = sys.argv[1], sys.argv[2]

    try:
        password = PASSWORD_FILE.read_text().strip()
    except Exception as e:
        print(f"[email] cannot read password file: {e}", file=sys.stderr)
        sys.exit(3)

    msg = EmailMessage()
    msg["From"] = FROM_ADDR
    msg["To"] = TO_ADDR
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=15) as s:
            s.login(FROM_ADDR, password)
            s.send_message(msg)
        print(f"[email] sent: {subject}")
    except Exception as e:
        print(f"[email] send failed: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
