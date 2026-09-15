# Real-video unit contract

| Input | Original synthetic | Original real | Corrected real |
|---|---|---|---|
| Dynamic decoder correction | (dynamic-static) / tracker sigma | axis-projected pixels | projected pixels / RGB noise estimate |
| Static decoder correction | (static-anchor) / tracker sigma | axis-projected pixels | projected pixels / same noise estimate |
| Pooled32 | decoder hidden states | same hidden-state computation | unchanged |
| Long bridge | not part of synthetic readout | deceleration / initial speed, camera dummy | numerator and denominator both rescaled; ratio invariant |
| Scalar long proxy | not part of synthetic readout | deceleration / initial speed | same invariance |

The real noise estimate is 1.4826 times the median absolute deviation of eight
tracked centers about a quadratic trend, floored at one resized-image pixel.
It is an observation-only proxy, not measured ground-truth tracking error.
The raw 33-video scores reproduce exactly before applying this correction.
Normalizing units does not match camera geometry, decoder training distribution,
or the physical meaning of the score. In particular the score is not friction.
The long bridge uses all 152 frames; its ratio already cancels pixel units and
must not be divided by noise again after forming the ratio.
