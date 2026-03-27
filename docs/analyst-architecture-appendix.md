# Analyst Architecture Appendix

This appendix explains the system context behind the analyst prompt. It is supporting context, not the primary execution contract.

## Four Layers

1. `Data layer`
   - collects raw macro, price, SEC, and geo-event data
   - stores snapshots point-in-time correctly

2. `Causal layer`
   - normalizes entities and relations
   - derives themes, regimes, analogs, pricing discipline, cross-asset confirmation, trade readiness, and confidence states

3. `Quant layer`
   - engineers features
   - runs the LightGBM ranker
   - produces model interpretation artifacts

4. `Analyst layer`
   - reads the structured packet
   - explains the setup
   - produces research or decision output
   - can be downgraded by the critic or trust gate

## Reasoning Order

1. facts
2. causal state
3. source and narrative quality
4. analog context
5. market confirmation
6. model context
7. actionability
8. uncertainty and falsification

## Operational Notes

- `CrossAssetConfirmationState` includes energy, vol, credit, rates, FX, sector-flow, and peer confirmation.
- `PricingDisciplineState` is about confirmation, lateness, extension, and follow-through.
- `TradeReadinessState` is about actionability, not truth.
- Missing cross-asset confirmation should weaken pricing confidence more than interpretation.
- Critic logic and trust-tier gating can downgrade `decision` to `research_only`.
