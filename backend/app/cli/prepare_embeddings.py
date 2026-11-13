import argparse
from pathlib import Path

from app.ai.embeddings.backend import EmbeddingModelError
from app.ai.embeddings.service import EmbeddingService
from app.core.config import get_settings
from app.schemas.embeddings import EmbeddingConfig


def main():
    parser = argparse.ArgumentParser(description='Prepara e valida o encoder fixo no cache persistente.')
    parser.add_argument('--cache-dir', type=Path, default=get_settings().embeddings_cache_dir)
    parser.add_argument('--offline', action='store_true', help='Valida o cache sem baixar arquivos.')
    args = parser.parse_args()
    config = EmbeddingConfig(cache_dir=args.cache_dir, allow_download=not args.offline)
    try:
        vector = EmbeddingService(config).embed_query('Validação do encoder local.')
    except EmbeddingModelError as error:
        parser.exit(1, f'{error}\n')
    print(f'Encoder validado: {len(vector.vector)} dimensões; revisão {config.revision}.')


if __name__ == '__main__':
    main()
