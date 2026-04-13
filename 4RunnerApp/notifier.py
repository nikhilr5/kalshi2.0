"""
Email notification for order fills.

Uses Gmail SMTP with an App Password. Set these environment variables:
    GMAIL_USER=your_email@gmail.com
    GMAIL_APP_PASSWORD=your_16_char_app_password

To create an App Password:
    1. Go to https://myaccount.google.com/apppasswords
    2. Select "Mail" and "Other (custom name)"
    3. Copy the 16-character password
    4. export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
"""

import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


RECIPIENT = "nikhil.richard84@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def send_fill_notification(ticker: str, side: str, action: str,
                           price: float, quantity: int, order_id: str):
    """Send email notification when an order gets filled.
    Runs in a background thread so it doesn't block the GUI."""
    thread = threading.Thread(
        target=_send_email,
        args=(ticker, side, action, price, quantity, order_id),
        daemon=True,
    )
    thread.start()


def _send_email(ticker: str, side: str, action: str,
                price: float, quantity: int, order_id: str):
    """Actually send the email. Runs on background thread."""
    gmail_user = os.environ.get("GMAIL_USER", "")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")

    if not gmail_user or not gmail_pass:
        print("[EMAIL] GMAIL_USER or GMAIL_APP_PASSWORD not set, skipping notification")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"Kalshi Fill: {action.upper()} {side.upper()} {ticker}"
    body = f"""
Order Filled!

Time: {now}
Ticker: {ticker}
Side: {side}
Action: {action}
Price: ${price:.2f}
Quantity: {quantity}
Order ID: {order_id}
Total Value: ${price * quantity:.2f}

— 4RunnerApp
"""

    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = RECIPIENT
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, RECIPIENT, msg.as_string())

        print(f"[EMAIL] Fill notification sent for {ticker}")
    except Exception as e:
        print(f"[EMAIL] Failed to send: {e}")
