from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _load_beneficiaries() -> list[dict]:
    default_path = Path(__file__).parent.parent / "data" / "beneficiaries.json"
    path = Path(os.getenv("BENEFICIARIES_FILE", str(default_path)))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BENEFICIARIES = _load_beneficiaries()


class Beneficiary(BaseModel):
    id: str = Field(description="Internal beneficiary ID.")
    full_name: str = Field(description="Full official name.")
    account_number: str = Field(description="Account number.")
    bank_name: str = Field(description="Bank name.")


class TransactionIntent(BaseModel):
    intent: Literal["send_money", "buy_airtime", "pay_bill", "check_balance", "other"] = Field(
        description="The type of financial action the user wants to perform."
    )
    amount: float = Field(
        description="Numeric amount. Interpret shorthand: 5k=5000, 10k=10000, 1m=1000000."
    )
    currency: str = Field(
        default="NGN",
        description="ISO 4217 currency code. Default to NGN if not specified.",
    )
    recipient: Beneficiary | None = Field(
        default=None,
        description=(
            "The resolved beneficiary from the database. "
            "Match on aliases or full_name. Null if no match found."
        ),
    )

    @classmethod
    def system_prompt(cls) -> str:
        db_context = "Available Beneficiaries:\n" + json.dumps(BENEFICIARIES, indent=2)

        return (
            "You are a financial data extraction assistant processing voice-transcribed text.\n"
            "The input may contain speech recognition errors.\n\n"
            f"{db_context}\n\n"
            "Rules:\n"
            "- Extract the transaction details from the user's spoken command.\n"
            "- Default currency to NGN if not stated.\n"
            "- Map currencies and slangs to ISO 4217 codes:\n"
            "    'naira' -> NGN\n"
            "    'dollars', 'bucks', '$' -> USD\n"
            "    'pounds', 'quid', '£', 'p', 'pence' -> GBP\n"
            "    'euros', '€' -> EUR\n"
            "- If the amount is in pence/cents (like '50p'), represent it as a decimal (e.g. 0.50).\n"
            "- Interpret amount shorthand and spoken forms:\n"
            "    '5k', 'five k', 'five thousand' -> 5000\n"
            "    '20k', 'twenty k', 'twenty thousand' -> 20000\n"
            "    '1m', 'one million' -> 1000000\n"
            "    'five hundred' -> 500\n"
            "- Correct obvious speech recognition errors for names "
            "(e.g. 'gyms' likely means 'james').\n"
            "- Match the recipient against the beneficiaries list using aliases or full_name.\n"
            "- Return the full beneficiary object for the recipient field. "
            "Null if no match found.\n"
            "- Return only the JSON object, no explanation."
        )
