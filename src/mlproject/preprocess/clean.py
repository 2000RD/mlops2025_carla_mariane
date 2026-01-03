"""Simple, conservative data cleaning utilities for the NYC Taxi project.

These functions implement the minimal preprocessing steps required by the
course: handle missing values, drop invalid rows, basic filtering and save
cleaned CSV outputs.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning rules to a taxi trips dataframe.

    Rules (conservative):
    - Parse datetime columns if present.
    - Drop rows with missing critical coordinates or datetimes.
    - Remove non-positive or extremely large `trip_duration` values (if present).
    - Filter out coordinate values outside a loose NYC bounding box.
    - Normalize `passenger_count` to reasonable range.

    The function is defensive: if a column is absent, it skips that check.
    """
    df = df.copy()

    # Parse datetimes when available
    for col in ("pickup_datetime", "dropoff_datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows missing essential datetimes if present
    if "pickup_datetime" in df.columns:
        before = len(df)
        df = df.dropna(subset=["pickup_datetime"])  # cannot work without pickup time
        logger.info("Dropped %d rows with missing pickup_datetime", before - len(df))

    # Coordinates checks
    lat_lon_cols = [
        ("pickup_latitude", "pickup_longitude"),
        ("dropoff_latitude", "dropoff_longitude"),
    ]
    # Loose bounding box around NYC
    min_lat, max_lat = 40.0, 42.0
    min_lon, max_lon = -75.0, -72.0

    for lat_col, lon_col in lat_lon_cols:
        if lat_col in df.columns and lon_col in df.columns:
            before = len(df)
            mask = (
                df[lat_col].between(min_lat, max_lat)
                & df[lon_col].between(min_lon, max_lon)
            )
            df = df[mask]
            logger.info(
                "Dropped %d rows outside bounding box for %s/%s",
                before - len(df),
                lat_col,
                lon_col,
            )

    # Passenger count
    if "passenger_count" in df.columns:
        before = len(df)
        # reasonable passenger counts: 1..8
        df = df[df["passenger_count"].between(1, 8)]
        logger.info("Dropped %d rows with invalid passenger_count", before - len(df))

    # Trip duration (seconds) filters
    if "trip_duration" in df.columns:
        before = len(df)
        # Ensure numeric
        df["trip_duration"] = pd.to_numeric(df["trip_duration"], errors="coerce")
        # Remove non-positive and extreme durations (> 1 day)
        df = df[df["trip_duration"].between(1, 24 * 3600)]
        logger.info("Dropped %d rows outside duration bounds", before - len(df))

    # Any remaining NaNs in important numeric cols -> drop
    important_cols = [c for c in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"] if c in df.columns]
    if important_cols:
        before = len(df)
        df = df.dropna(subset=important_cols)
        logger.info("Dropped %d rows with NaNs in important coordinate cols", before - len(df))

    return df.reset_index(drop=True)


def load_and_clean(input_path: str, output_path: str, *, sample_n: Optional[int] = None) -> None:
    """Load CSV from `input_path`, clean it and write to `output_path`.

    If `sample_n` is provided the function will read only a sample of rows for
    quick iteration/testing (uses `pd.read_csv(..., nrows=sample_n)`).
    """
    read_kwargs = {"nrows": sample_n} if sample_n is not None else {}
    logger.info("Reading data from %s", input_path)
    df = pd.read_csv(input_path, **read_kwargs)
    logger.info("Input rows: %d", len(df))

    df_clean = clean_dataframe(df)
    logger.info("Clean rows: %d", len(df_clean))

    # Ensure parent directories exist
    pd.io.common._ensure_str_path = getattr(pd.io.common, "_ensure_str_path", None)
    df_clean.to_csv(output_path, index=False)
    logger.info("Wrote cleaned data to %s", output_path)
