from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    entity_type: str
    entity_id: str
    entity_name: str
    source: str = "v1"


def slugify_entity(value: str) -> str:
    return value.strip().lower().replace("&", "and").replace(" ", "_").replace("-", "_")


ROUTE_ALIASES = {
    "bab_el_mandeb": "bab_el_mandeb",
    "red_sea": "red_sea",
    "suez": "suez_canal",
    "suez_canal": "suez_canal",
    "strait_of_hormuz": "strait_of_hormuz",
}

COUNTRY_ALIASES = {
    "us": "united_states",
    "usa": "united_states",
    "uk": "united_kingdom",
    "uae": "united_arab_emirates",
}

COMMODITY_ALIASES = {
    "crude": "oil",
    "brent": "oil",
    "lng": "lng",
    "natgas": "gas",
    "natural_gas": "gas",
    "jet_fuel": "jet_fuel",
}


def canonicalize_entity_name(entity_type: str, raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    slug = slugify_entity(text)
    if entity_type == "route":
        return ROUTE_ALIASES.get(slug, slug)
    if entity_type == "country":
        return COUNTRY_ALIASES.get(slug, slug)
    if entity_type == "commodity":
        return COMMODITY_ALIASES.get(slug, slug)
    return slug


def normalize_event_entities(record: dict[str, Any]) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []

    for entity_name in record.get("affected_entities", []) or []:
        normalized = canonicalize_entity_name("company_or_asset", entity_name)
        if normalized:
            entities.append(
                CanonicalEntity(
                    entity_type="company_or_asset",
                    entity_id=normalized,
                    entity_name=str(entity_name).strip(),
                )
            )

    for sector_name in record.get("affected_sectors", []) or []:
        normalized = canonicalize_entity_name("sector", sector_name)
        if normalized:
            entities.append(
                CanonicalEntity(
                    entity_type="sector",
                    entity_id=normalized,
                    entity_name=str(sector_name).strip(),
                )
            )

    for commodity_name in record.get("affected_commodities", []) or []:
        normalized = canonicalize_entity_name("commodity", commodity_name)
        if normalized:
            entities.append(
                CanonicalEntity(
                    entity_type="commodity",
                    entity_id=normalized,
                    entity_name=str(commodity_name).strip(),
                )
            )

    geography = record.get("geography") or record.get("region")
    if geography:
        normalized = canonicalize_entity_name("route", geography)
        entities.append(
            CanonicalEntity(
                entity_type="route",
                entity_id=normalized,
                entity_name=str(geography).strip(),
            )
        )

    event_family = record.get("event_family")
    if event_family:
        normalized = canonicalize_entity_name("event_type", event_family)
        entities.append(
            CanonicalEntity(
                entity_type="event_type",
                entity_id=normalized,
                entity_name=str(event_family).strip(),
            )
        )

    deduped: dict[tuple[str, str], CanonicalEntity] = {}
    for entity in entities:
        deduped[(entity.entity_type, entity.entity_id)] = entity
    return list(deduped.values())


def build_event_relations(
    *,
    event_id: str,
    record: dict[str, Any],
    normalized_entities: list[CanonicalEntity],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    theme = str(record.get("theme") or "")
    topic = str(record.get("topic") or "")
    region = str(record.get("region") or record.get("geography") or "")
    market_relevance = float(record.get("market_relevance", 0.0) or 0.0)

    for entity in normalized_entities:
        relation_type = "affects"
        if entity.entity_type == "route":
            relation_type = "targets_route"
        elif entity.entity_type == "commodity":
            relation_type = "affects_commodity"
        elif entity.entity_type == "sector":
            relation_type = "affects_sector"

        relations.append(
            {
                "event_id": event_id,
                "source_type": "event",
                "source_id": event_id,
                "relation_type": relation_type,
                "target_type": entity.entity_type,
                "target_id": entity.entity_id,
                "confidence": float(record.get("confidence", 0.0) or 0.0),
                "evidence_text": topic,
                "theme": theme or None,
                "region": region or None,
                "market_relevance": market_relevance,
                "transform_loaded_at": record.get("transform_loaded_at"),
            }
        )

    dependency_markers = []
    if theme == "oil_supply_risk":
        dependency_markers.append(
            ("sector", "consumer_discretionary", "depends_on", "commodity", "jet_fuel")
        )
    if theme == "gas_supply_risk":
        dependency_markers.append(("region", "europe", "depends_on", "commodity", "gas"))
    if theme == "shipping_stress":
        dependency_markers.append(
            ("sector", "consumer_discretionary", "depends_on", "route", "red_sea")
        )
        dependency_markers.append(("sector", "industrials", "depends_on", "route", "red_sea"))

    for source_type, source_id, relation_type, target_type, target_id in dependency_markers:
        relations.append(
            {
                "event_id": event_id,
                "source_type": source_type,
                "source_id": source_id,
                "relation_type": relation_type,
                "target_type": target_type,
                "target_id": target_id,
                "confidence": float(record.get("confidence", 0.0) or 0.0),
                "evidence_text": topic,
                "theme": theme or None,
                "region": region or None,
                "market_relevance": market_relevance,
                "transform_loaded_at": record.get("transform_loaded_at"),
            }
        )

    return relations


def default_entity_reference_frames() -> dict[str, pd.DataFrame]:
    now = pd.Timestamp.utcnow()
    return {
        "entity_company": pd.DataFrame(
            [
                {
                    "entity_id": "xom",
                    "ticker": "XOM",
                    "name": "Exxon Mobil",
                    "sector": "Energy",
                    "country": "US",
                    "source": "v1",
                },
                {
                    "entity_id": "cvx",
                    "ticker": "CVX",
                    "name": "Chevron",
                    "sector": "Energy",
                    "country": "US",
                    "source": "v1",
                },
                {
                    "entity_id": "aal",
                    "ticker": "AAL",
                    "name": "American Airlines",
                    "sector": "Consumer Discretionary",
                    "country": "US",
                    "source": "v1",
                },
                {
                    "entity_id": "dal",
                    "ticker": "DAL",
                    "name": "Delta Air Lines",
                    "sector": "Consumer Discretionary",
                    "country": "US",
                    "source": "v1",
                },
                {
                    "entity_id": "lmt",
                    "ticker": "LMT",
                    "name": "Lockheed Martin",
                    "sector": "Industrials",
                    "country": "US",
                    "source": "v1",
                },
            ]
        ),
        "entity_country": pd.DataFrame(
            [
                {
                    "entity_id": "united_states",
                    "country_code": "US",
                    "name": "United States",
                    "region": "North America",
                    "source": "v1",
                },
                {
                    "entity_id": "europe",
                    "country_code": "EU",
                    "name": "Europe",
                    "region": "Europe",
                    "source": "v1",
                },
                {
                    "entity_id": "saudi_arabia",
                    "country_code": "SA",
                    "name": "Saudi Arabia",
                    "region": "Middle East",
                    "source": "v1",
                },
            ]
        ),
        "entity_region": pd.DataFrame(
            [
                {"entity_id": "middle_east", "name": "Middle East", "source": "v1"},
                {"entity_id": "red_sea", "name": "Red Sea", "source": "v1"},
                {"entity_id": "europe", "name": "Europe", "source": "v1"},
            ]
        ),
        "entity_commodity": pd.DataFrame(
            [
                {"entity_id": "oil", "symbol": "CL", "name": "Oil", "source": "v1"},
                {"entity_id": "gas", "symbol": "NG", "name": "Natural Gas", "source": "v1"},
                {"entity_id": "lng", "symbol": "LNG", "name": "LNG", "source": "v1"},
                {"entity_id": "jet_fuel", "symbol": "JET", "name": "Jet Fuel", "source": "v1"},
            ]
        ),
        "entity_sector": pd.DataFrame(
            [
                {"entity_id": "energy", "name": "Energy", "source": "v1"},
                {"entity_id": "industrials", "name": "Industrials", "source": "v1"},
                {
                    "entity_id": "consumer_discretionary",
                    "name": "Consumer Discretionary",
                    "source": "v1",
                },
                {
                    "entity_id": "information_technology",
                    "name": "Information Technology",
                    "source": "v1",
                },
            ]
        ),
        "entity_route": pd.DataFrame(
            [
                {"entity_id": "red_sea", "name": "Red Sea", "source": "v1"},
                {"entity_id": "bab_el_mandeb", "name": "Bab el-Mandeb", "source": "v1"},
                {"entity_id": "strait_of_hormuz", "name": "Strait of Hormuz", "source": "v1"},
                {"entity_id": "suez_canal", "name": "Suez Canal", "source": "v1"},
            ]
        ),
        "entity_event_type": pd.DataFrame(
            [
                {"entity_id": "shipping_disruption", "name": "shipping_disruption", "source": "v1"},
                {"entity_id": "refinery_outage", "name": "refinery_outage", "source": "v1"},
                {"entity_id": "military_escalation", "name": "military_escalation", "source": "v1"},
                {"entity_id": "cyber_attack", "name": "cyber_attack", "source": "v1"},
            ]
        ),
        "entity_policy_action": pd.DataFrame(
            [
                {"entity_id": "sanctions", "name": "Sanctions", "source": "v1"},
                {"entity_id": "policy_relief", "name": "Policy Relief", "source": "v1"},
                {"entity_id": "export_ban_tariff", "name": "Export Ban or Tariff", "source": "v1"},
            ]
        ),
        "dependency_markers": pd.DataFrame(
            [
                {
                    "source_type": "sector",
                    "source_id": "consumer_discretionary",
                    "relation_type": "depends_on",
                    "target_type": "commodity",
                    "target_id": "jet_fuel",
                    "weight": 0.9,
                    "rationale": "Airlines are highly sensitive to jet fuel prices.",
                    "source": "v1",
                    "transform_loaded_at": now,
                },
                {
                    "source_type": "region",
                    "source_id": "europe",
                    "relation_type": "depends_on",
                    "target_type": "commodity",
                    "target_id": "gas",
                    "weight": 0.7,
                    "rationale": (
                        "European industry remains sensitive to pipeline and LNG gas flows."
                    ),
                    "source": "v1",
                    "transform_loaded_at": now,
                },
                {
                    "source_type": "sector",
                    "source_id": "consumer_discretionary",
                    "relation_type": "depends_on",
                    "target_type": "route",
                    "target_id": "red_sea",
                    "weight": 0.6,
                    "rationale": (
                        "Retail and travel activity are exposed to "
                        "shipping throughput and reroutes."
                    ),
                    "source": "v1",
                    "transform_loaded_at": now,
                },
            ]
        ),
    }
