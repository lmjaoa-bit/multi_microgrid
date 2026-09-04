from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
LOCATIONS = {
    "mg1": {"name": "Nanjing", "latitude": 32.0603, "longitude": 118.7969},
    "mg2": {"name": "Yancheng", "latitude": 33.3474, "longitude": 120.1636},
    "mg3": {"name": "Suzhou", "latitude": 31.2989, "longitude": 120.5853},
}
SPLITS = ("train", "val", "test")


def fetch_hourly_weather(location: dict[str, float | str]) -> tuple[pd.DataFrame, dict[str, str]]:
    query = urlencode({
        "parameters": "ALLSKY_SFC_SW_DWN,WS50M",
        "community": "RE",
        "longitude": location["longitude"],
        "latitude": location["latitude"],
        "start": "20180101",
        "end": "20181231",
        "format": "JSON",
        "time-standard": "LST",
    })
    url = f"{BASE_URL}?{query}"
    with urlopen(url, timeout=120) as response:  # noqa: S310 - fixed NASA endpoint
        payload = json.load(response)
    parameters = payload["properties"]["parameter"]
    ghi = pd.Series(parameters["ALLSKY_SFC_SW_DWN"], name="ghi_w_per_m2", dtype=float)
    wind = pd.Series(parameters["WS50M"], name="wind_speed_mps", dtype=float)
    frame = pd.concat([ghi, wind], axis=1)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    frame.index.name = "timestamp"
    frame = frame.replace(-999.0, np.nan)
    if frame.isna().any().any():
        raise ValueError(f"NASA POWER returned missing weather for {location['name']}")
    units = {
        "ghi_w_per_m2": str(payload["parameters"]["ALLSKY_SFC_SW_DWN"]["units"]),
        "wind_speed_mps": str(payload["parameters"]["WS50M"]["units"]),
    }
    return frame, {"url": url, **units}


def main() -> int:
    weather: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, str]] = {}
    for microgrid_id, location in LOCATIONS.items():
        print(f"Downloading {microgrid_id} weather: {location['name']}", flush=True)
        weather[microgrid_id], sources[microgrid_id] = fetch_hourly_weather(location)

    outputs: dict[str, str] = {}
    for split in SPLITS:
        source = PROJECT_ROOT / "data" / f"uk_microgrid_2018_{split}.csv"
        target = PROJECT_ROOT / "data" / f"spatial_market_2018_{split}.csv"
        frame = pd.read_csv(source)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        for microgrid_id, local_weather in weather.items():
            aligned = local_weather.reindex(frame["timestamp"])
            if aligned.isna().any().any():
                raise ValueError(f"Weather does not cover every {split} timestamp for {microgrid_id}")
            frame[f"{microgrid_id}_ghi_w_per_m2"] = aligned["ghi_w_per_m2"].to_numpy()
            frame[f"{microgrid_id}_wind_speed_mps"] = aligned["wind_speed_mps"].to_numpy()
        frame.to_csv(target, index=False, date_format="%Y-%m-%d %H:%M:%S")
        outputs[split] = str(target.relative_to(PROJECT_ROOT))

    metadata = {
        "description": "Spatially heterogeneous, same-calendar NASA POWER weather joined to the existing load/market-index splits.",
        "year": 2018,
        "time_standard": "LST",
        "solar_parameter": "ALLSKY_SFC_SW_DWN",
        "wind_parameter": "WS50M",
        "locations": LOCATIONS,
        "sources": sources,
        "outputs": outputs,
    }
    metadata_path = PROJECT_ROOT / "data" / "spatial_market_2018_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
