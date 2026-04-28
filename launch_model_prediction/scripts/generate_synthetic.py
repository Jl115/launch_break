"""Generate synthetic historical JSON data for testing."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

base_date = datetime(2024, 1, 1)
while base_date.weekday() != 0:
    base_date += timedelta(days=1)

WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]

date_iter = base_date
for week in range(25):
    week_start = date_iter
    closed = set(random.sample(WEEKDAYS, random.choice([0, 1])))
    schedule = {}
    for d in WEEKDAYS:
        if d in closed:
            schedule[d] = {
                "status": "geschlossen",
                "menu": None,
                "signups": None,
                "summary_bottom": None,
            }
        else:
            schedule[d] = {
                "status": "open",
                "menu": None,
                "signups": None,
                "summary_bottom": {
                    "Erw": random.randint(3, 15),
                    "Ki": random.randint(2, 20),
                    "MA": random.randint(2, 12),
                    "MA-Ki": random.randint(0, 5),
                },
            }
    totals = {"Erw": 0, "Ki": 0, "MA": 0, "MA-Ki": 0}
    for d in schedule.values():
        if d.get("summary_bottom"):
            for k in totals:
                totals[k] += d["summary_bottom"].get(k, 0)
    data = {
        "header": {
            "facility": "Familienzentrum Bern KITAS MURIFELD",
            "document_type": "Menu - Anmeldeliste",
            "date": f"{week_start.day}.{week_start.month}.{week_start.year}",
        },
        "schedule": schedule,
        "weekly_totals": totals,
    }
    fname = f"{week_start.year}-{week_start.month:02d}-{week_start.day:02d}.json"
    out = Path("data/json") / fname
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    date_iter += timedelta(days=7)

print(f"Generated {len(list(Path('data/json').glob('*.json')))} JSON files")
