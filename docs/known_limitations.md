# Known limitations

## Belt period metadata

BeltMap can handle cyclic belt coordinates when the physical belt period is supplied. If no period is supplied, the reconstructed map height is inferred from the observed sequence and should be treated as finite support rather than as a guaranteed physical circumference.

The lower-level rendering APIs support finite-support rendering. The full image driver still has remaining period-metadata propagation work tracked in issue #28.

Until that work is complete:

- Supply `BELT_PERIOD_PX` when the physical belt period is known.
- Be careful when using recurrent-artifact filtering without a supplied period.
- Treat maps built without a supplied period as sequence-support reconstructions rather than calibrated cyclic belt models.
