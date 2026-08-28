"""Where and how the weekly pick is submitted.

The pick is emailed to the organiser, so the app composes that message rather
than leaving it to be retyped each week. Both values are overridable by
environment variable for anyone running this in another pool.
"""

from __future__ import annotations

import os
from urllib.parse import quote, urlencode

GMAIL_COMPOSE = "https://mail.google.com/mail/?view=cm&fs=1"


def contest_email() -> str:
    """Address the weekly pick is submitted to."""
    return os.getenv("CONTEST_EMAIL", "premierPicksCompetition@gmail.com")


def entrant_name() -> str:
    """Name the pick is submitted under."""
    return os.getenv("CONTEST_ENTRANT", "Franco Solari")


def compose_pick_email(
    contest_week: int,
    team: str,
    opponent: str,
) -> dict[str, str]:
    """Build the submission email for a committed pick.

    The body names the picked team first regardless of venue — "I pick X to
    beat Y" reads the same whether X is at home or away, and the organiser
    needs the selection, not the fixture orientation.
    """
    subject = f"GW{contest_week} - {entrant_name()}"
    body = f"I pick {team} to beat {opponent}"
    recipient = contest_email()
    query = urlencode({"to": recipient, "su": subject, "body": body}, quote_via=quote)
    mailto_query = urlencode({"subject": subject, "body": body}, quote_via=quote)
    return {
        "to": recipient,
        "subject": subject,
        "body": body,
        "gmail_url": f"{GMAIL_COMPOSE}&{query}",
        "mailto_url": f"mailto:{recipient}?{mailto_query}",
    }
