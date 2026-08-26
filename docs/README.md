# Presentation

`Smart-EV-Charging-Deck.pdf` — a 5-slide overview of the project (1920 × 1080, 16:9).

| slide | content |
|---|---|
| 1 | Title, network scale, the two audiences |
| 2 | The dataset and how it is simulated, station × hour utilisation heatmap |
| 3 | The three models, held-out accuracy, feature families |
| 4 | The linear program and the 28 % peak reduction |
| 5 | Driver app, operator dashboard, stack |

Every number and both charts are generated from the project's own artifacts —
nothing is hand-typed.

## Rebuilding it

```bash
python -m src.models.train      # refresh artifacts/reports/metrics.json first, if needed
python docs/build_deck.py       # regenerates Smart-EV-Charging-Deck.html
```

Then print the HTML to PDF from any Chromium browser (the page is already sized to
20in × 11.25in with `@page`):

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=docs/Smart-EV-Charging-Deck.pdf docs/Smart-EV-Charging-Deck.html
```

`deck_data1.json` / `deck_data2.json` hold the extracted chart data (utilisation
heatmap, load profiles) so the deck can be rebuilt without rerunning the forecaster.
