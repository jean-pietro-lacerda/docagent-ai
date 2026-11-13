import json
import resource
import sys


MAX_TEXT_CHARACTERS = 1_000_000


def inspect_pdf(path: str, *, extract_text: bool = False) -> dict:
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
    pymupdf.TOOLS.mupdf_warnings(reset=True)

    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                return {"error": "invalid"}

            if document.needs_pass or document.is_encrypted or (document.metadata or {}).get("encryption"):
                return {"error": "encrypted"}

            if document.is_repaired:
                return {"error": "corrupt"}

            pages = document.page_count

            if pages < 1:
                return {"error": "empty_pages"}

            if pages > 20:
                return {"error": "pages"}

            texts = []
            total_characters = 0

            for index in range(pages):
                page = document.load_page(index)

                if extract_text:
                    try:
                        text = page.get_text("text", sort=True)
                    except (RuntimeError, ValueError, pymupdf.mupdf.FzErrorBase):
                        texts.append(None)
                        continue

                    total_characters += len(text)

                    if total_characters > MAX_TEXT_CHARACTERS:
                        return {"error": "text_limit"}

                    texts.append(text)

            if document.is_repaired or pymupdf.TOOLS.mupdf_warnings(reset=True):
                return {"error": "corrupt"}

            if extract_text:
                return {"pages": pages, "page_texts": texts}

            return {"pages": pages}
    except MemoryError:
        return {"error": "resources"}
    except Exception:
        return {"error": "corrupt"}


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(json.dumps(inspect_pdf(sys.argv[1], extract_text=sys.argv[2:] == ["extract"])))
