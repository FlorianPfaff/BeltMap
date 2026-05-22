# Known limitations

## Belt period metadata

BeltMap can handle cyclic belt coordinates when the physical belt period is supplied. If no period is supplied, the reconstructed map height is inferred from the observed sequence and should be treated as finite support rather than as a guaranteed physical circumference.

The lower-level rendering and residual APIs support finite-support rendering. The period-state helper layer also distinguishes physical model periods from finite map support and validates reused metadata for obvious incompatibilities, such as stale `belt_map_height_px`, missing periodic period values, or period values that disagree with the loaded map height.

New period-state metadata should include `belt_map_height_px`, `model_period_px`, `belt_period_known`, `belt_map_periodic`, and `belt_period_state_source` together. The height field is intentionally redundant with the array shape so reuse paths can detect stale metadata that belongs to a different `belt_map.npy`.

The full image driver still has remaining period-metadata propagation work tracked in issue #28. In particular, driver orchestration must still be updated to construct its motion model from the resolved physical period state rather than treating every loaded map height as a physical cyclic period.

Until that driver work is complete:

- Supply `BELT_PERIOD_PX` when the physical belt period is known.
- Avoid recurrent-artifact filtering for maps built without a known physical period unless you have verified the driver path uses compatible period metadata.
- Treat maps built without a supplied period as sequence-support reconstructions rather than calibrated cyclic belt models.
- When reusing maps, keep `belt_map.npy`, `metadata.json`, and any phase or recurrent-artifact outputs from the same run together; the helper layer now rejects several stale or contradictory metadata combinations, but the full driver still needs end-to-end period-state wiring.