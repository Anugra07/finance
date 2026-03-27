from __future__ import annotations

from dataclasses import asdict, dataclass

GRAPH_VERSION = "v2"
ANALOG_MODEL_VERSION = "v2"
CAUSAL_STATE_VERSION = "v2"
INTERPRETATION_VERSION = "v2"
REASONING_SCHEMA_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class VersionMetadata:
    graph_version: str = GRAPH_VERSION
    analog_model_version: str = ANALOG_MODEL_VERSION
    causal_state_version: str = CAUSAL_STATE_VERSION
    interpretation_version: str = INTERPRETATION_VERSION
    reasoning_schema_version: str = REASONING_SCHEMA_VERSION

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def default_version_metadata() -> VersionMetadata:
    return VersionMetadata()
