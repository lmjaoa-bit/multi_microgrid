from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Three nearby nodes in the Yancheng-area regional scenario.  They are kept as
# distinct coordinates and recorded in metadata. NASA POWER can quantize nearby
# points to the same source grid cell; the script never fabricates differences.
LOCATIONS = {
    "mg1": {
        "name": "Yancheng Industrial Node",
        "latitude": 33.38,
        "longitude": 120.13,
    },
    "mg2": {
        "name": "Dafeng Commercial Node",
        "latitude": 33.20,
        "longitude": 120.46,
    },
    "mg3": {
        "name": "Yandu Residential Node",
        "latitude": 33.34,
        "longitude": 119.98,
    },
}
SPLITS = ("train", "val", "test")


def fetch_hourly_weather(location: dict[str, float | str]) -> tuple[pd.DataFrame, dict[str, str]]:
    query = urlencode({
        "parameters": "ALLSKY_SFC_SW_DWN,WS50M,T2M",
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
    temperature = pd.Series(parameters["T2M"], name="temperature_c", dtype=float)
    frame = pd.concat([ghi, wind, temperature], axis=1)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    frame.index.name = "timestamp"
    frame = frame.replace(-999.0, np.nan)
    if frame.isna().any().any():
        raise ValueError(f"NASA POWER returned missing weather for {location['name']}")
    units = {
        "ghi_w_per_m2": str(payload["parameters"]["ALLSKY_SFC_SW_DWN"]["units"]),
        "wind_speed_mps": str(payload["parameters"]["WS50M"]["units"]),
        "temperature_c": str(payload["parameters"]["T2M"]["units"]),
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
                raise ValueError(
                    f"Weather does not cover every {split} timestamp for {microgrid_id}"
                )
            frame[f"{microgrid_id}_ghi_w_per_m2"] = aligned["ghi_w_per_m2"].to_numpy()
            frame[f"{microgrid_id}_wind_speed_mps"] = aligned["wind_speed_mps"].to_numpy()
            frame[f"{microgrid_id}_temperature_c"] = aligned["temperature_c"].to_numpy()
        frame.to_csv(target, index=False, date_format="%Y-%m-%d %H:%M:%S")
        outputs[split] = str(target.relative_to(PROJECT_ROOT))

    metadata = {
        "description": (
            "Yancheng-area regional scenario using same-calendar NASA POWER solar, "
            "wind and 2 m temperature joined to the existing load/market-index splits. "
            "Nearby points may resolve to the same NASA source grid cell; no synthetic "
            "weather difference is introduced by this builder."
        ),
        "year": 2018,
        "time_standard": "LST",
        "solar_parameter": "ALLSKY_SFC_SW_DWN",
        "wind_parameter": "WS50M",
        "temperature_parameter": "T2M",
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
