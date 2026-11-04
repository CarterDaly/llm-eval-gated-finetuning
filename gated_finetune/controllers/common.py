"""Request helpers shared by the controllers."""
from flask import jsonify, request
from pydantic import BaseModel, ValidationError


def missing_fields_message(fields: list[str]) -> str:
    return "Missing required fields: {}".format(", ".join(sorted(fields)))


def parse_body(model_class: type[BaseModel]):
    """Parse and validate the JSON request body against a Pydantic model.

    Returns (model_instance, None) on success or (None, error_response) on failure.
    """
    data = request.get_json(silent=True)
    required = [n for n, f in model_class.model_fields.items() if f.is_required()]
    if not isinstance(data, dict):
        return None, (jsonify({"error": missing_fields_message(required)}), 400)
    try:
        return model_class.model_validate(data), None
    except ValidationError as exc:
        errors = exc.errors()
        missing = [str(err["loc"][0]) for err in errors if err.get("type") == "missing"]
        if missing:
            return None, (jsonify({"error": missing_fields_message(missing)}), 400)
        if any("umap" in err.get("loc", ()) for err in errors):
            return None, (jsonify({"error": "Invalid UMAP configuration"}), 400)
        return None, (jsonify({"error": "Invalid request body"}), 400)
