import re


def normalize_username(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", normalized):
        raise ValueError("Username deve ter de 3 a 64 caracteres: letras ASCII, números, ponto, hífen ou sublinhado.")
    return normalized
