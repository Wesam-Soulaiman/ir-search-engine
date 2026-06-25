from typing import Any, Optional


TRUE_VALUES = {
    True,
    1,
    "1",
    "true",
    "yes",
    "on",
}

FALSE_VALUES = {
    False,
    0,
    "0",
    "false",
    "no",
    "off",
    "",
    None,
}


def parse_boolean(
    value: Any,
    field_name: str,
) -> bool:
    normalized_value = (
        value.strip().lower()
        if isinstance(value, str)
        else value
    )

    if normalized_value in TRUE_VALUES:
        return True

    if normalized_value in FALSE_VALUES:
        return False

    raise ValueError(
        f"'{field_name}' must be a boolean value."
    )


def parse_integer(
    value: Any,
    field_name: str,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if value is None:
        if default is None:
            raise ValueError(
                f"'{field_name}' is required."
            )

        result = int(default)

    else:
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'{field_name}' must be an integer."
            )

    if minimum is not None and result < minimum:
        raise ValueError(
            f"'{field_name}' must be at least "
            f"{minimum}."
        )

    if maximum is not None and result > maximum:
        raise ValueError(
            f"'{field_name}' must not exceed "
            f"{maximum}."
        )

    return result


def parse_float(
    value: Any,
    field_name: str,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if value is None:
        if default is None:
            raise ValueError(
                f"'{field_name}' is required."
            )

        result = float(default)

    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'{field_name}' must be numeric."
            )

    if minimum is not None and result < minimum:
        raise ValueError(
            f"'{field_name}' must be at least "
            f"{minimum}."
        )

    if maximum is not None and result > maximum:
        raise ValueError(
            f"'{field_name}' must not exceed "
            f"{maximum}."
        )

    return result