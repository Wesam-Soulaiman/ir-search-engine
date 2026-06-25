import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Load the project-level .env file.
load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=False,
)


def env_boolean(
    name: str,
    default: bool = False,
) -> bool:
    """
    Read a strict boolean environment variable.
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = (
        raw_value.strip().lower()
    )

    if normalized_value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized_value in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return False

    raise RuntimeError(
        f"Environment variable '{name}' "
        "must contain a boolean value."
    )


def env_list(
    name: str,
    default: str = "",
) -> List[str]:
    """
    Read a comma-separated environment variable.
    """
    raw_value = os.getenv(
        name,
        default,
    )

    return [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]


def env_path(
    name: str,
    default: Path,
) -> Path:
    """
    Read a path environment variable.

    Relative paths are resolved from the project root.
    """
    raw_value = os.getenv(name)

    if raw_value:
        path = Path(
            raw_value
        ).expanduser()
    else:
        path = Path(default)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-local-development-only",
)

DEBUG = env_boolean(
    "DJANGO_DEBUG",
    default=True,
)

ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    default="localhost,127.0.0.1",
)


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "rest_framework",
    "corsheaders",

    "preprocessing",
    "indexing",
    "retrieval",
    "evaluation",
    "query_refinement",
]


MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "backend.urls"

TEMPLATES = [
    {
        "BACKEND": (
            "django.template.backends.django."
            "DjangoTemplates"
        ),
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                (
                    "django.template.context_processors."
                    "request"
                ),
                (
                    "django.contrib.auth."
                    "context_processors.auth"
                ),
                (
                    "django.contrib.messages."
                    "context_processors.messages"
                ),
            ],
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"


DATABASE_PATH = env_path(
    "DJANGO_DATABASE_PATH",
    BASE_DIR / "db.sqlite3",
)

DATABASES = {
    "default": {
        "ENGINE": (
            "django.db.backends.sqlite3"
        ),
        "NAME": DATABASE_PATH,
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"

USE_I18N = True
USE_TZ = True


STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = (
    "django.db.models.BigAutoField"
)


CORS_ALLOW_ALL_ORIGINS = env_boolean(
    "DJANGO_CORS_ALLOW_ALL_ORIGINS",
    default=False,
)

CORS_ALLOWED_ORIGINS = env_list(
    "DJANGO_CORS_ALLOWED_ORIGINS",
    default=(
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    ),
)

CORS_ALLOW_CREDENTIALS = False

CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    default=(
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    ),
)


# Shared project paths used by future services.
DATA_DIR = env_path(
    "IR_DATA_DIR",
    PROJECT_ROOT / "data",
)

INDEXES_DIR = env_path(
    "IR_INDEXES_DIR",
    PROJECT_ROOT / "indexes",
)

REPORTS_DIR = env_path(
    "IR_REPORTS_DIR",
    PROJECT_ROOT / "reports",
)

ARTIFACTS_DIR = env_path(
    "IR_ARTIFACTS_DIR",
    PROJECT_ROOT / "artifacts",
)


# When enabled, Hugging Face and Transformers are prohibited from
# downloading models. They must load from local cache/artifacts.
IR_OFFLINE_MODE = env_boolean(
    "IR_OFFLINE_MODE",
    default=False,
)

if IR_OFFLINE_MODE:
    os.environ.setdefault(
        "HF_HUB_OFFLINE",
        "1",
    )

    os.environ.setdefault(
        "TRANSFORMERS_OFFLINE",
        "1",
    )


REST_FRAMEWORK = {
    "DEFAULT_PARSER_CLASSES": [
        (
            "rest_framework.parsers."
            "JSONParser"
        ),
    ],
    "DEFAULT_RENDERER_CLASSES": [
        (
            "rest_framework.renderers."
            "JSONRenderer"
        ),
    ],
}


LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": (
                "{asctime} | {levelname} | "
                "{name} | {message}"
            ),
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": (
                "logging.StreamHandler"
            ),
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": [
            "console",
        ],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "django.server": {
            "handlers": [
                "console",
            ],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "retrieval": {
            "handlers": [
                "console",
            ],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "evaluation": {
            "handlers": [
                "console",
            ],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}

CORPUS_DATABASE_PATH = env_path(
    "IR_CORPUS_DATABASE_PATH",
    ARTIFACTS_DIR / "database" / "corpus.sqlite3",
)