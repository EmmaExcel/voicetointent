# Demo

Five test phrases that show what this system can do. Run them with the CLI to see the full pipeline without needing an audio file.

---

## Setup

Make sure your `.env` is configured and your chosen LLM is running, then:

```bash
python cli.py extract "<phrase>" --schema financial
```

---

## Test Cases

### 1. Alias resolution

```bash
python cli.py extract "Send 5k to mum"
```

Expected: `intent=send_money`, `amount=5000`, recipient resolved to Jane Doe (alias: mum).

---

### 2. Amount shorthand

```bash
python cli.py extract "Transfer twenty thousand naira to my brother"
```

Expected: `intent=send_money`, `amount=20000`, recipient resolved to John Smith (alias: brother).

---

### 3. Speech error correction

```bash
python cli.py extract "Send 3k to gyms"
```

Expected: recipient resolved to James O. — "gyms" is a known alias for James, a common speech recognition mishearing.

---

### 4. Bill payment

```bash
python cli.py extract "Pay my electricity bill, fifteen thousand"
```

Expected: `intent=pay_bill`, `amount=15000`, recipient resolved to Ikeja Electric (aliases: nepa, light, electricity, power).

---

### 5. Airtime top-up

```bash
python cli.py extract "Buy 2k airtime for myself"
```

Expected: `intent=buy_airtime`, `amount=2000`, recipient resolved to MTN Mobile (alias: myself, my phone, airtime).

---

## What to Look For

- **Alias matching**: the LLM finds the right beneficiary from a colloquial name or alias, not just an exact name match.
- **Amount normalization**: spoken shorthand ("5k", "twenty thousand", "1m") is consistently converted to a numeric integer.
- **Intent classification**: `send_money`, `buy_airtime`, `pay_bill`, `check_balance`, `other` — correctly derived from phrasing.
- **Confidence score**: when run against an audio file (via `process` or `listen`), a `confidence` value from Whisper indicates transcription quality.

---

## Switching Providers

Change `LLM_PROVIDER` in `.env` to any supported value and re-run the same commands. The extraction logic and output format stays identical.

```bash
# .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
```
