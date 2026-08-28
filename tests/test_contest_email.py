"""The weekly pick is submitted by email, so the app composes that message."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from epl_prediction_optimizer.config.contest import compose_pick_email


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("CONTEST_EMAIL", raising=False)
    monkeypatch.delenv("CONTEST_ENTRANT", raising=False)
    monkeypatch.delenv("CONTEST_GMAIL_ACCOUNT", raising=False)


def test_the_email_goes_to_the_organiser_with_the_round_and_entrant() -> None:
    email = compose_pick_email(2, "Coventry City", "Hull City")
    assert email["to"] == "premierPicksCompetition@gmail.com"
    assert email["subject"] == "GW2 - Franco Solari"
    assert email["body"] == "I pick Coventry City to beat Hull City"


def test_the_picked_team_is_named_first_even_when_it_is_the_away_side() -> None:
    email = compose_pick_email(1, "Manchester United", "Hull City")
    assert email["body"] == "I pick Manchester United to beat Hull City"


def test_the_gmail_link_carries_the_composed_message() -> None:
    email = compose_pick_email(2, "Coventry City", "Hull City")
    parsed = urlparse(email["gmail_url"])
    query = parse_qs(parsed.query)
    assert parsed.netloc == "mail.google.com"
    assert query["view"] == ["cm"]
    assert query["to"] == ["premierPicksCompetition@gmail.com"]
    assert query["su"] == ["GW2 - Franco Solari"]
    assert query["body"] == ["I pick Coventry City to beat Hull City"]


def test_a_mailto_fallback_is_offered_for_the_default_mail_app() -> None:
    email = compose_pick_email(2, "Coventry City", "Hull City")
    assert email["mailto_url"].startswith("mailto:premierPicksCompetition@gmail.com?")
    query = parse_qs(urlparse(email["mailto_url"]).query)
    assert query["subject"] == ["GW2 - Franco Solari"]


def test_teams_with_spaces_and_ampersands_survive_the_link(monkeypatch) -> None:
    email = compose_pick_email(7, "Brighton & Hove Albion", "Nottingham Forest")
    query = parse_qs(urlparse(email["gmail_url"]).query)
    assert query["body"] == ["I pick Brighton & Hove Albion to beat Nottingham Forest"]


def test_another_pool_can_override_recipient_and_name(monkeypatch) -> None:
    monkeypatch.setenv("CONTEST_EMAIL", "someone@example.com")
    monkeypatch.setenv("CONTEST_ENTRANT", "Sam Doe")
    email = compose_pick_email(3, "Arsenal", "Everton")
    assert email["to"] == "someone@example.com"
    assert email["subject"] == "GW3 - Sam Doe"


def test_without_an_account_the_link_uses_whatever_gmail_defaults_to() -> None:
    url = compose_pick_email(2, "Coventry City", "Hull City")["gmail_url"]
    assert url.startswith("https://mail.google.com/mail/?")
    assert "/u/" not in url


def test_a_numeric_index_selects_the_account_in_the_path() -> None:
    """Gmail picks the account from the digit in /mail/u/<n>/ and nothing else."""
    email = compose_pick_email(2, "Coventry City", "Hull City", account="1")
    assert email["gmail_url"].startswith("https://mail.google.com/mail/u/1/?")
    assert email["account"] == "1"


def test_an_address_is_passed_as_authuser_not_as_a_path_segment() -> None:
    """An address in the path is ignored and silently falls back to the default."""
    email = compose_pick_email(2, "Coventry City", "Hull City", account="franco@gmail.com")
    assert "/mail/u/franco" not in email["gmail_url"]
    assert "authuser=franco%40gmail.com" in email["gmail_url"]


def test_the_account_can_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONTEST_GMAIL_ACCOUNT", "2")
    email = compose_pick_email(2, "Coventry City", "Hull City")
    assert "/mail/u/2/" in email["gmail_url"]


def test_an_explicit_account_beats_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONTEST_GMAIL_ACCOUNT", "2")
    email = compose_pick_email(2, "Coventry City", "Hull City", account="3")
    assert "/mail/u/3/" in email["gmail_url"]


def test_pinning_does_not_disturb_the_message(monkeypatch) -> None:
    plain = compose_pick_email(2, "Coventry City", "Hull City")
    pinned = compose_pick_email(2, "Coventry City", "Hull City", account="1")
    assert plain["subject"] == pinned["subject"]
    assert plain["body"] == pinned["body"]
    assert plain["to"] == pinned["to"]
