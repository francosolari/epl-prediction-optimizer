"""Where and how the weekly pick is submitted.

The pick is emailed to the organiser, so the app composes that message rather
than leaving it to be retyped each week. Both values are overridable by
environment variable for anyone running this in another pool.
"""

from __future__ import annotations

import os
from urllib.parse import quote, urlencode

GMAIL_BASE = "https://mail.google.com/mail"


def contest_email() -> str:
    """Address the weekly pick is submitted to."""
    return os.getenv("CONTEST_EMAIL", "premierPicksCompetition@gmail.com")


def entrant_name() -> str:
    """Name the pick is submitted under."""
    return os.getenv("CONTEST_ENTRANT", "Franco Solari")


def gmail_compose_url(subject: str, body: str, recipient: str, account: str | None) -> str:
    """Build a Gmail compose link, pinned to an account when one is known.

    Without an account Gmail composes from whichever session the browser has
    as default, which is unpredictable when several are signed in. Putting the
    address in the /mail/u/ path makes Gmail resolve that specific account
    instead, so the message cannot go out from the wrong one.
    """
    query = urlencode(
        {"view": "cm", "fs": "1", "to": recipient, "su": subject, "body": body},
        quote_via=quote,
    )
    inbox = f"u/{quote(account)}/" if account else ""
    return f"{GMAIL_BASE}/{inbox}?{query}"


def compose_pick_email(
    contest_week: int,
    team: str,
    opponent: str,
    account: str | None = None,
) -> dict[str, str]:
    """Build the submission email for a committed pick.

    The body names the picked team first regardless of venue — "I pick X to
    beat Y" reads the same whether X is at home or away, and the organiser
    needs the selection, not the fixture orientation.
    """
    subject = f"GW{contest_week} - {entrant_name()}"
    body = f"I pick {team} to beat {opponent}"
    recipient = contest_email()
    account = account or os.getenv("CONTEST_GMAIL_ACCOUNT") or None
    mailto_query = urlencode({"subject": subject, "body": body}, quote_via=quote)
    return {
        "to": recipient,
        "subject": subject,
        "body": body,
        "account": account or "",
        "gmail_url": gmail_compose_url(subject, body, recipient, account),
        "mailto_url": f"mailto:{recipient}?{mailto_query}",
    }
