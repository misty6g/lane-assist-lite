# Demo assets

All stills and the short clip in this folder are **procedurally generated**
by `python -m lane_assist generate`. They are not scraped dashcam frames.

| File | Scene |
| --- | --- |
| `straight.png` | Daylight, centered lane, no curvature |
| `curve.png` | Gentle right curve with a small lateral offset |
| `dusk.png` | Lower sun, cooler sky, slight left offset |
| `night.png` | Dark scene with headlight bloom |
| `clip.mp4` | 3-second sequence with a sinusoidal lateral weave |

Regenerate after changing `lane_assist.synthetic`:

```bash
python -m lane_assist generate -o data/samples
```
