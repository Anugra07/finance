from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

GEO_ENERGY_THEMES = [
    "oil_supply_risk",
    "gas_supply_risk",
    "shipping_stress",
    "sanctions_pressure",
    "grid_stress",
    "defense_escalation",
    "cyber_infra_risk",
    "industrial_metal_tightness",
    "policy_relief",
    "macro_demand_softness",
]

EVENT_FAMILY_THEME_MAP: dict[str, str] = {
    "military_escalation": "defense_escalation",
    "sanctions": "sanctions_pressure",
    "shipping_disruption": "shipping_stress",
    "chokepoint_disruption": "shipping_stress",
    "cyber_attack": "cyber_infra_risk",
    "refinery_outage": "oil_supply_risk",
    "pipeline_outage": "gas_supply_risk",
    "power_grid_failure": "grid_stress",
    "opec_statement": "oil_supply_risk",
    "lng_disruption": "gas_supply_risk",
    "weather_climate_shock": "grid_stress",
    "export_ban_tariff": "industrial_metal_tightness",
    "protest_regime_instability": "macro_demand_softness",
    "policy_shock": "policy_relief",
}

THEME_ALIASES: dict[str, str] = {
    "oil": "oil_supply_risk",
    "gas": "gas_supply_risk",
    "shipping": "shipping_stress",
    "sanctions": "sanctions_pressure",
    "defense": "defense_escalation",
    "cyber": "cyber_infra_risk",
    "metals": "industrial_metal_tightness",
    "policy": "policy_relief",
    "cyber_infrastructure_risk": "cyber_infra_risk",
}

THEME_TO_FEATURE_COLUMN: dict[str, str] = {
    "oil_supply_risk": "oil_supply_risk_1d",
    "gas_supply_risk": "gas_supply_risk_1d",
    "shipping_stress": "shipping_stress_1d",
    "sanctions_pressure": "sanctions_pressure_1d",
    "defense_escalation": "defense_escalation_1d",
    "grid_stress": "grid_stress_1d",
    "cyber_infra_risk": "cyber_infra_risk_1d",
    "policy_relief": "policy_relief_prob_1d",
}

DEFAULT_SECTOR_THEME_EXPOSURES = [
    {
        "sector": "Communication Services",
        "theme": "cyber_infra_risk",
        "exposure": -0.25,
        "transmission_label": "network_fragility",
        "rationale": (
            "Communications networks and digital advertising demand weaken "
            "when cyber incidents widen."
        ),
    },
    {
        "sector": "Communication Services",
        "theme": "policy_relief",
        "exposure": 0.15,
        "transmission_label": "regulatory_tailwind",
        "rationale": "Policy easing can support advertising demand and platform multiples.",
    },
    {
        "sector": "Consumer Discretionary",
        "theme": "oil_supply_risk",
        "exposure": -0.75,
        "transmission_label": "fuel_cost_pressure",
        "rationale": (
            "Air travel, autos, and retail absorb higher fuel and logistics costs quickly."
        ),
    },
    {
        "sector": "Consumer Discretionary",
        "theme": "shipping_stress",
        "exposure": -0.60,
        "transmission_label": "import_delay",
        "rationale": (
            "Retail inventories and discretionary goods are vulnerable to reroutes and port stress."
        ),
    },
    {
        "sector": "Consumer Discretionary",
        "theme": "macro_demand_softness",
        "exposure": -0.55,
        "transmission_label": "demand_slowdown",
        "rationale": "Discretionary demand rolls over fastest when growth expectations soften.",
    },
    {
        "sector": "Consumer Staples",
        "theme": "shipping_stress",
        "exposure": -0.25,
        "transmission_label": "inventory_resilience",
        "rationale": "Staples are defensive but still face freight and import friction.",
    },
    {
        "sector": "Consumer Staples",
        "theme": "macro_demand_softness",
        "exposure": 0.20,
        "transmission_label": "defensive_rotation",
        "rationale": "Staples often benefit on a relative basis during demand slowdowns.",
    },
    {
        "sector": "Energy",
        "theme": "oil_supply_risk",
        "exposure": 0.95,
        "transmission_label": "commodity_upside",
        "rationale": (
            "Upstream and service names benefit when supply disruptions lift crude pricing."
        ),
    },
    {
        "sector": "Energy",
        "theme": "gas_supply_risk",
        "exposure": 0.70,
        "transmission_label": "lng_tightness",
        "rationale": "Gas and LNG-linked names gain pricing power during supply stress.",
    },
    {
        "sector": "Energy",
        "theme": "sanctions_pressure",
        "exposure": 0.35,
        "transmission_label": "supply_repricing",
        "rationale": "Sanctions can reprice supply chains toward domestic or friendly producers.",
    },
    {
        "sector": "Financials",
        "theme": "macro_demand_softness",
        "exposure": -0.40,
        "transmission_label": "credit_cycle",
        "rationale": "Banks and credit providers face lower demand and higher stress in slowdowns.",
    },
    {
        "sector": "Financials",
        "theme": "policy_relief",
        "exposure": 0.30,
        "transmission_label": "liquidity_support",
        "rationale": "Policy relief can tighten spreads and support financial conditions.",
    },
    {
        "sector": "Health Care",
        "theme": "macro_demand_softness",
        "exposure": 0.25,
        "transmission_label": "defensive_quality",
        "rationale": "Health care tends to be a relative defensive winner when growth weakens.",
    },
    {
        "sector": "Industrials",
        "theme": "defense_escalation",
        "exposure": 0.75,
        "transmission_label": "defense_demand",
        "rationale": (
            "Defense contractors and select aerospace suppliers benefit from "
            "higher conflict intensity."
        ),
    },
    {
        "sector": "Industrials",
        "theme": "shipping_stress",
        "exposure": -0.35,
        "transmission_label": "supply_chain_friction",
        "rationale": "Global industrial supply chains are exposed to reroutes and port congestion.",
    },
    {
        "sector": "Industrials",
        "theme": "grid_stress",
        "exposure": 0.40,
        "transmission_label": "infrastructure_spend",
        "rationale": (
            "Grid equipment, engineering, and backup-power suppliers gain on "
            "infrastructure response."
        ),
    },
    {
        "sector": "Information Technology",
        "theme": "cyber_infra_risk",
        "exposure": -0.45,
        "transmission_label": "cyber_supply_fragility",
        "rationale": (
            "Semis and hardware face geopolitical and infrastructure risk even "
            "as security subsectors benefit."
        ),
    },
    {
        "sector": "Information Technology",
        "theme": "industrial_metal_tightness",
        "exposure": -0.30,
        "transmission_label": "input_cost_pressure",
        "rationale": "Semiconductors and hardware are sensitive to critical mineral tightness.",
    },
    {
        "sector": "Information Technology",
        "theme": "policy_relief",
        "exposure": 0.35,
        "transmission_label": "capex_support",
        "rationale": (
            "Policy easing and industrial support often revive tech risk appetite and spending."
        ),
    },
    {
        "sector": "Materials",
        "theme": "industrial_metal_tightness",
        "exposure": 0.80,
        "transmission_label": "metal_beta",
        "rationale": (
            "Materials respond directly to copper, uranium, and critical mineral tightness."
        ),
    },
    {
        "sector": "Materials",
        "theme": "macro_demand_softness",
        "exposure": -0.45,
        "transmission_label": "cyclical_drag",
        "rationale": "Materials fade when end-demand expectations weaken.",
    },
    {
        "sector": "Real Estate",
        "theme": "grid_stress",
        "exposure": -0.20,
        "transmission_label": "power_costs",
        "rationale": (
            "Power instability raises operating risk for real estate and data-center landlords."
        ),
    },
    {
        "sector": "Real Estate",
        "theme": "policy_relief",
        "exposure": 0.40,
        "transmission_label": "rate_tailwind",
        "rationale": "Real estate is highly sensitive to easing policy and lower financing costs.",
    },
    {
        "sector": "Utilities",
        "theme": "grid_stress",
        "exposure": 0.45,
        "transmission_label": "resilience_spend",
        "rationale": (
            "Utilities and grid operators benefit from resilience capex when grid stress rises."
        ),
    },
    {
        "sector": "Utilities",
        "theme": "gas_supply_risk",
        "exposure": -0.35,
        "transmission_label": "fuel_input_risk",
        "rationale": (
            "Gas-tightness can pressure fuel costs for utilities with weak supply security."
        ),
    },
]

DEFAULT_SOLUTION_MAPPINGS = [
    {
        "theme": "oil_supply_risk",
        "solution_type": "beneficiary",
        "label": "Domestic producers and oilfield services",
        "beneficiary_sector": "Energy",
        "hedge_role": "inflation_hedge",
        "rationale": (
            "Tighter oil balances support upstream producers, storage, and service providers."
        ),
    },
    {
        "theme": "gas_supply_risk",
        "solution_type": "beneficiary",
        "label": "LNG exporters and storage infrastructure",
        "beneficiary_sector": "Energy",
        "hedge_role": "supply_hedge",
        "rationale": "Gas shortages reprice LNG, storage, and secure domestic supply chains.",
    },
    {
        "theme": "shipping_stress",
        "solution_type": "stabilizer",
        "label": "Alternate-route logistics and port infrastructure",
        "beneficiary_sector": "Industrials",
        "hedge_role": "supply_chain_resilience",
        "rationale": "Rerouting and port upgrades absorb chokepoint stress and freight repricing.",
    },
    {
        "theme": "sanctions_pressure",
        "solution_type": "beneficiary",
        "label": "Friendly-shore suppliers and compliance infrastructure",
        "beneficiary_sector": "Industrials",
        "hedge_role": "sanctions_compliance",
        "rationale": (
            "Sanctions expansions reward compliant alternative supply chains "
            "and verification layers."
        ),
    },
    {
        "theme": "grid_stress",
        "solution_type": "infrastructure_response",
        "label": "Grid equipment, transformers, and backup power",
        "beneficiary_sector": "Industrials",
        "hedge_role": "resilience_buildout",
        "rationale": (
            "Grid failures accelerate spending on transformers, switchgear, and backup systems."
        ),
    },
    {
        "theme": "defense_escalation",
        "solution_type": "beneficiary",
        "label": "Defense contractors and surveillance systems",
        "beneficiary_sector": "Industrials",
        "hedge_role": "conflict_hedge",
        "rationale": (
            "Escalation raises demand for defense platforms, ISR, and replenishment cycles."
        ),
    },
    {
        "theme": "cyber_infra_risk",
        "solution_type": "beneficiary",
        "label": "Cybersecurity and incident-response vendors",
        "beneficiary_sector": "Information Technology",
        "hedge_role": "cyber_hedge",
        "rationale": (
            "Infrastructure attacks push security budgets toward endpoint, "
            "network, and SOC tooling."
        ),
    },
    {
        "theme": "industrial_metal_tightness",
        "solution_type": "beneficiary",
        "label": "Copper, uranium, and recycling chains",
        "beneficiary_sector": "Materials",
        "hedge_role": "commodity_hedge",
        "rationale": (
            "Critical metal scarcity rewards miners, recyclers, and substitute supply chains."
        ),
    },
    {
        "theme": "policy_relief",
        "solution_type": "policy_action",
        "label": "Policy-easing beneficiaries",
        "beneficiary_sector": "Real Estate",
        "hedge_role": "duration_tailwind",
        "rationale": (
            "Easing policy supports duration-sensitive sectors and high-multiple growth assets."
        ),
    },
]

DEFAULT_INDUSTRY_THEME_EXPOSURES = [
    {
        "industry": "Passenger Airlines",
        "theme": "oil_supply_risk",
        "exposure": -0.95,
        "transmission_label": "jet_fuel_beta",
        "rationale": "Airlines are direct losers when crude and jet-fuel stress rise.",
    },
    {
        "industry": "Passenger Airlines",
        "theme": "shipping_stress",
        "exposure": -0.35,
        "transmission_label": "travel_chain_friction",
        "rationale": (
            "Travel demand and airport logistics can soften when regional disruptions widen."
        ),
    },
    {
        "industry": "Aerospace & Defense",
        "theme": "defense_escalation",
        "exposure": 0.95,
        "transmission_label": "defense_budget_beta",
        "rationale": "Defense primes and suppliers benefit most directly from escalation cycles.",
    },
    {
        "industry": "Oil & Gas Exploration & Production",
        "theme": "oil_supply_risk",
        "exposure": 1.00,
        "transmission_label": "upstream_beta",
        "rationale": "E&P names have the highest direct beta to tighter oil balances.",
    },
    {
        "industry": "Oil & Gas Storage & Transportation",
        "theme": "gas_supply_risk",
        "exposure": 0.70,
        "transmission_label": "midstream_tightness",
        "rationale": "Pipelines, LNG, and storage assets benefit from stressed gas logistics.",
    },
    {
        "industry": "Semiconductors",
        "theme": "cyber_infra_risk",
        "exposure": -0.50,
        "transmission_label": "fab_fragility",
        "rationale": (
            "Chip supply chains are highly exposed to infrastructure and geopolitical shocks."
        ),
    },
    {
        "industry": "Semiconductors",
        "theme": "industrial_metal_tightness",
        "exposure": -0.45,
        "transmission_label": "critical_input_costs",
        "rationale": "Semiconductor production depends on tight critical-mineral supply chains.",
    },
    {
        "industry": "Electronic Equipment & Instruments",
        "theme": "grid_stress",
        "exposure": 0.45,
        "transmission_label": "backup_power_demand",
        "rationale": (
            "Electrical equipment suppliers see stronger demand during grid hardening cycles."
        ),
    },
    {
        "industry": "Electric Utilities",
        "theme": "grid_stress",
        "exposure": 0.55,
        "transmission_label": "resilience_capex",
        "rationale": (
            "Utilities participate directly in resilience and grid modernization spending."
        ),
    },
    {
        "industry": "Systems Software",
        "theme": "cyber_infra_risk",
        "exposure": 0.30,
        "transmission_label": "security_spend",
        "rationale": "Security-linked software budgets rise when cyber infrastructure risk widens.",
    },
]

DEFAULT_STOCK_THEME_EXPOSURES = [
    {
        "ticker": "AAL",
        "theme": "oil_supply_risk",
        "exposure": -1.00,
        "transmission_label": "airline_fuel_cost",
        "rationale": "American Airlines is highly exposed to fuel-driven margin compression.",
    },
    {
        "ticker": "DAL",
        "theme": "oil_supply_risk",
        "exposure": -0.95,
        "transmission_label": "airline_fuel_cost",
        "rationale": "Delta is directly exposed to oil and jet-fuel stress.",
    },
    {
        "ticker": "UAL",
        "theme": "oil_supply_risk",
        "exposure": -0.95,
        "transmission_label": "airline_fuel_cost",
        "rationale": "United Airlines is a direct loser when oil supply stress rises.",
    },
    {
        "ticker": "COP",
        "theme": "oil_supply_risk",
        "exposure": 0.95,
        "transmission_label": "upstream_oil_beta",
        "rationale": "ConocoPhillips is a high-beta upstream oil beneficiary.",
    },
    {
        "ticker": "CVX",
        "theme": "oil_supply_risk",
        "exposure": 0.90,
        "transmission_label": "integrated_oil_beta",
        "rationale": "Chevron benefits from rising oil prices and tighter supply balances.",
    },
    {
        "ticker": "XOM",
        "theme": "oil_supply_risk",
        "exposure": 0.90,
        "transmission_label": "integrated_oil_beta",
        "rationale": "Exxon gains when oil supply risk lifts crude pricing.",
    },
    {
        "ticker": "LNG",
        "theme": "gas_supply_risk",
        "exposure": 1.00,
        "transmission_label": "lng_export_beta",
        "rationale": (
            "Cheniere is one of the cleanest public beneficiaries of gas supply tightness."
        ),
    },
    {
        "ticker": "LMT",
        "theme": "defense_escalation",
        "exposure": 1.00,
        "transmission_label": "prime_contractor_beta",
        "rationale": (
            "Lockheed Martin is directly levered to defense escalation and replenishment demand."
        ),
    },
    {
        "ticker": "NOC",
        "theme": "defense_escalation",
        "exposure": 0.95,
        "transmission_label": "prime_contractor_beta",
        "rationale": "Northrop Grumman benefits from higher defense urgency and ISR demand.",
    },
    {
        "ticker": "RTX",
        "theme": "defense_escalation",
        "exposure": 0.90,
        "transmission_label": "prime_contractor_beta",
        "rationale": "RTX benefits from missile, defense-electronics, and replenishment cycles.",
    },
    {
        "ticker": "CRWD",
        "theme": "cyber_infra_risk",
        "exposure": 0.90,
        "transmission_label": "cyber_budget_beta",
        "rationale": "CrowdStrike should benefit when cyber infrastructure risk intensifies.",
    },
    {
        "ticker": "PANW",
        "theme": "cyber_infra_risk",
        "exposure": 0.90,
        "transmission_label": "cyber_budget_beta",
        "rationale": (
            "Palo Alto Networks benefits from incident response and security budget acceleration."
        ),
    },
    {
        "ticker": "ETN",
        "theme": "grid_stress",
        "exposure": 0.85,
        "transmission_label": "grid_equipment_beta",
        "rationale": "Eaton is a clear beneficiary of grid hardening and backup power investment.",
    },
    {
        "ticker": "HUBB",
        "theme": "grid_stress",
        "exposure": 0.80,
        "transmission_label": "grid_equipment_beta",
        "rationale": (
            "Hubbell benefits from transmission, distribution, and grid modernization spend."
        ),
    },
    {
        "ticker": "NVDA",
        "theme": "industrial_metal_tightness",
        "exposure": -0.40,
        "transmission_label": "critical_input_risk",
        "rationale": (
            "NVIDIA faces indirect hardware and infrastructure sensitivity to critical minerals."
        ),
    },
]


def normalize_theme(theme: str | None) -> str | None:
    if theme is None:
        return None
    lowered = theme.strip().lower().replace(" ", "_")
    return THEME_ALIASES.get(lowered, lowered)


def infer_theme(event_family: str | None, theme: str | None = None) -> str | None:
    normalized_theme = normalize_theme(theme)
    if normalized_theme:
        return normalized_theme
    if not event_family:
        return None
    return EVENT_FAMILY_THEME_MAP.get(event_family.strip().lower().replace(" ", "_"))


def default_sector_theme_exposures() -> pd.DataFrame:
    frame = pd.DataFrame(DEFAULT_SECTOR_THEME_EXPOSURES)
    frame["source"] = "hand_coded_v1"
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    return frame


def default_industry_theme_exposures() -> pd.DataFrame:
    frame = pd.DataFrame(DEFAULT_INDUSTRY_THEME_EXPOSURES)
    frame["source"] = "hand_coded_v1"
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    return frame


def default_stock_theme_exposures() -> pd.DataFrame:
    frame = pd.DataFrame(DEFAULT_STOCK_THEME_EXPOSURES)
    frame["source"] = "hand_coded_v1"
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    return frame


def default_solution_mappings() -> pd.DataFrame:
    frame = pd.DataFrame(DEFAULT_SOLUTION_MAPPINGS)
    frame["source"] = "hand_coded_v1"
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    return frame
