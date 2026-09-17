from schemas.financial import TransactionIntent

SCHEMA_REGISTRY: dict[str, type] = {
    "financial": TransactionIntent,
}


def get_schema(name: str):
    if name not in SCHEMA_REGISTRY:
        available = ", ".join(SCHEMA_REGISTRY.keys())
        raise KeyError(
            f"Schema '{name}' not found. Available schemas: {available}"
        )
    return SCHEMA_REGISTRY[name]


def list_schemas() -> list[str]:
    return list(SCHEMA_REGISTRY.keys())
