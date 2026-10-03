"""Conversion d'un JSON schema de task.yaml en modele Pydantic.

Le meme modele sert partout : validation des exemples generes (F2), des sorties
du professeur, et des sorties du SLM pendant l'evaluation (taux de JSON valide).
On couvre volontairement un sous-ensemble de JSON schema : objets plats ou
legerement imbriques, enums, listes. C'est ce que couvre le perimetre v1
(classification, extraction de champs, routage).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .errors import SchemaError

_PRIMITIVES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _annotation(spec: dict[str, Any], name: str) -> Any:
    """Traduit une propriete JSON schema en annotation Python."""
    if "enum" in spec:
        values = spec["enum"]
        if not values:
            raise SchemaError(f"enum vide pour le champ '{name}'")
        return Literal[tuple(values)]  # type: ignore[valid-type]

    jtype = spec.get("type", "string")
    if isinstance(jtype, list):  # ex: ["string", "null"]
        non_null = [t for t in jtype if t != "null"]
        if not non_null:
            raise SchemaError(f"type invalide pour le champ '{name}'")
        inner = _annotation({**spec, "type": non_null[0]}, name)
        return Union[inner, None]  # noqa: UP007 - construit dynamiquement

    if jtype in _PRIMITIVES:
        return _PRIMITIVES[jtype]
    if jtype == "array":
        items = spec.get("items", {"type": "string"})
        return list[_annotation(items, f"{name}[]")]  # type: ignore[misc]
    if jtype == "object":
        return build_output_model(spec, name=f"{name.title().replace('_', '')}Model")
    raise SchemaError(f"type JSON schema non supporte pour '{name}' : {jtype!r}")


def build_output_model(schema: dict[str, Any], name: str = "TaskOutput") -> type[BaseModel]:
    """Construit un modele Pydantic a partir d'un JSON schema d'objet."""
    if schema.get("type") != "object":
        raise SchemaError("le schema racine doit etre de type 'object'")
    properties = schema.get("properties") or {}
    if not properties:
        raise SchemaError("le schema doit declarer au moins une propriete")
    required = set(schema.get("required", list(properties)))

    fields: dict[str, Any] = {}
    for field_name, spec in properties.items():
        if not isinstance(spec, dict):
            raise SchemaError(f"la propriete '{field_name}' doit etre un objet JSON schema")
        annotation = _annotation(spec, field_name)
        description = spec.get("description")
        if field_name in required:
            fields[field_name] = (annotation, Field(..., description=description))
        else:
            fields[field_name] = (annotation | None, Field(default=None, description=description))

    extra: Literal["forbid", "ignore"] = (
        "forbid" if schema.get("additionalProperties") is False else "ignore"
    )
    model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        name,
        __config__=ConfigDict(extra=extra),
        **fields,
    )
    return model


def schema_hint(schema: dict[str, Any]) -> str:
    """Rend le schema lisible pour un prompt, enums comprises."""
    lines: list[str] = []
    required = set(schema.get("required", list(schema.get("properties", {}))))
    for field_name, spec in (schema.get("properties") or {}).items():
        bits: list[str] = []
        if "enum" in spec:
            bits.append(
                "une valeur parmi "
                + " | ".join(json.dumps(v, ensure_ascii=False) for v in spec["enum"])
            )
        else:
            bits.append(str(spec.get("type", "string")))
        if spec.get("description"):
            bits.append(spec["description"])
        if field_name not in required:
            bits.append("optionnel")
        lines.append(f"- {field_name}: {', '.join(bits)}")
    return "\n".join(lines)


def enum_values(schema: dict[str, Any], field: str) -> list[Any]:
    """Valeurs possibles d'un champ enum, liste vide s'il n'en est pas un."""
    spec = (schema.get("properties") or {}).get(field, {})
    return list(spec.get("enum", []))


def primary_label_field(schema: dict[str, Any], override: str | None = None) -> str | None:
    """Champ de classification principal : le premier enum, sauf override explicite."""
    if override:
        return override
    for field_name, spec in (schema.get("properties") or {}).items():
        if isinstance(spec, dict) and spec.get("enum"):
            return str(field_name)
    return None


def label_fields(schema: dict[str, Any], override: list[str] | None = None) -> list[str]:
    """Tous les champs categoriels, utilises pour les F1 par classe."""
    if override:
        return list(override)
    return [
        f
        for f, spec in (schema.get("properties") or {}).items()
        if isinstance(spec, dict) and spec.get("enum")
    ]


def extract_json(text: str) -> Any:
    """Extrait le premier objet ou tableau JSON d'une reponse de LLM.

    Les modeles entourent souvent le JSON de texte ou de balises markdown. On
    nettoie plutot que de rejeter : c'est le role du validateur Pydantic
    ensuite de dire si le contenu est conforme.
    """
    if not text or not text.strip():
        raise ValueError("reponse vide")
    candidate = text.strip()

    block = _JSON_BLOCK.search(candidate)
    if block:
        candidate = block.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = min(
        (i for i in (candidate.find("{"), candidate.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError("aucun JSON trouve dans la reponse")
    opener = candidate[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : idx + 1])
    raise ValueError("JSON incomplet dans la reponse")


def validate_output(
    model: type[BaseModel], payload: Any
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Valide une sortie contre le schema. Retourne (ok, donnees normalisees, erreur)."""
    try:
        instance = model.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "<racine>"
        return False, None, f"{loc}: {first.get('msg')}"
    except Exception as exc:  # noqa: BLE001 - payload n'est meme pas un mapping
        return False, None, str(exc)
    return True, instance.model_dump(mode="json"), None


def literal_options(annotation: Any) -> list[Any]:
    """Valeurs d'un Literal, utile pour les tests et les rapports."""
    return list(get_args(annotation))
