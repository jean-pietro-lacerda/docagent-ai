import sys
import warnings
from getpass import GetPassWarning, getpass

from app.db.session import SessionLocal
from app.services.auth import AuthPersistenceError, create_local_user


def main() -> int:
    if len(sys.argv) != 1:
        print("Este comando não aceita argumentos; use os prompts interativos.", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("Execute este comando em um terminal interativo.", file=sys.stderr)
        return 2
    try:
        username = input("Username: ")
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            password = getpass("Senha: ")
            confirmation = getpass("Confirme a senha: ")
        if password != confirmation:
            print("As senhas não coincidem.", file=sys.stderr)
            return 1
        with SessionLocal() as session:
            create_local_user(session, username, password)
    except (ValueError, AuthPersistenceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (GetPassWarning, EOFError, KeyboardInterrupt):
        print("Criação cancelada; é necessário um terminal com senha sem eco.", file=sys.stderr)
        return 1
    print("Conta local criada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
