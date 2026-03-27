from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.events.normalization import normalize_event_frame


def test_event_normalization_emits_canonical_entities_and_relations() -> None:
    ingest_time = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {
                "topic": "Missile strike near Bab el-Mandeb",
                "event_family": "shipping_disruption",
                "theme": "shipping_stress",
                "region": "Red Sea",
                "geography": "Bab el-Mandeb",
                "severity": 0.8,
                "confidence": 0.9,
                "novelty": 0.7,
                "market_relevance": 0.9,
                "affected_commodities": ["crude", "lng"],
                "affected_sectors": ["Consumer Discretionary", "Industrials"],
                "affected_entities": ["AAL"],
                "raw_ref": "evt-1",
            }
        ]
    )

    normalized, entities, relations, source_assessment, narrative_risk, evidence_catalog = (
        normalize_event_frame(
        frame,
        source="test",
        ingest_time=ingest_time,
    )
    )

    assert len(normalized) == 1
    assert {"route", "commodity", "sector", "company_or_asset"} <= set(entities["entity_type"])
    assert "bab_el_mandeb" in entities["entity_id"].tolist()
    assert "oil" in entities["entity_id"].tolist()
    assert not relations.empty
    assert not source_assessment.empty
    assert not narrative_risk.empty
    assert not evidence_catalog.empty
    assert {"targets_route", "affects_commodity", "affects_sector"} <= set(
        relations["relation_type"]
    )
