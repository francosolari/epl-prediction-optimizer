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
    """Build a Gmail compose link, targeted at one account where possible.

    Gmail selects the account from the numeric index in the /mail/u/<n>/ path.
    An email address there is not honoured — it silently falls back to whatever
    account is signed in first, which is how a pick can go out from the wrong
    address with nothing on screen to show it. So a digit is used as the index,
    and an address is passed as authuser, which Google resolves where it can.

    The index is the only deterministic form. Open Gmail on the account you
    want and read it out of the URL: mail.google.com/mail/u/1/ is index 1.
    """
    params = {"view": "cm", "fs": "1", "to": recipient, "su": subject, "body": body}
    index = ""
    if account:
        if account.isdigit():
            index = f"u/{account}/"
        else:
            params["authuser"] = account
    query = urlencode(params, quote_via=quote)
    return f"{GMAIL_BASE}/{index}?{query}"


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
