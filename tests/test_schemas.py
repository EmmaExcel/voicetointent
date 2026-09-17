from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.financial import TransactionIntent, Beneficiary


VALID_BENEFICIARY = {
    "id": "ben_001",
    "full_name": "Jane Doe",
    "account_number": "0123456789",
    "bank_name": "Guaranty Trust Bank (GTB)",
}


def test_transaction_intent_basic():
    intent = TransactionIntent(
        intent="send_money",
        amount=5000,
        currency="NGN",
        recipient=VALID_BENEFICIARY,
    )
    assert intent.intent == "send_money"
    assert intent.amount == 5000
    assert intent.currency == "NGN"
    assert intent.recipient.full_name == "Jane Doe"


def test_transaction_intent_null_recipient():
    intent = TransactionIntent(intent="check_balance", amount=0, recipient=None)
    assert intent.recipient is None


def test_transaction_intent_default_currency():
    intent = TransactionIntent(intent="buy_airtime", amount=2000)
    assert intent.currency == "NGN"


def test_transaction_intent_invalid_intent():
    with pytest.raises(ValidationError):
        TransactionIntent(intent="invalid_action", amount=1000)


def test_transaction_intent_negative_amount():
    intent = TransactionIntent(intent="send_money", amount=-500)
    assert intent.amount == -500


def test_system_prompt_contains_beneficiaries():
    prompt = TransactionIntent.system_prompt()
    assert "beneficiar" in prompt.lower()
    assert "NGN" in prompt


def test_system_prompt_contains_rules():
    prompt = TransactionIntent.system_prompt()
    assert "5k" in prompt
    assert "5000" in prompt


def test_beneficiary_model():
    b = Beneficiary(**VALID_BENEFICIARY)
    assert b.id == "ben_001"
    assert b.account_number == "0123456789"
