import os

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

from django.conf import settings

from document_store.repository import (
    DocumentStoreRepository,
)


def main():
    repository = DocumentStoreRepository(
        settings.CORPUS_DATABASE_PATH
    )

    repository.initialize()

    print("=" * 70)
    print("Document store initialized successfully.")
    print(
        "Database path: "
        f"{repository.database_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()