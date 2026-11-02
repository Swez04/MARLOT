from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from math import asin, cos, isfinite, log1p, pi, radians, sin, sqrt
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_CATEGORY = "__MISSING__"

NUMERIC_FEATURES = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "account_txn_count_60s",
    "account_txn_count_5m",
    "seconds_since_prev_account_txn",
    "has_prev_account_txn",
    "log_amount_deviation",
    "has_account_amount_history",
    "account_amount_ratio",
    "account_amount_robust_zscore",
    "has_reliable_account_amount_history",
    "category_amount_deviation",
    "has_category_amount_history",
    "is_new_device_for_account",
    "location_changed",
    "geo_distance_km",
    "geo_travel_speed_kmph",
    "impossible_travel_flag",
    "has_geo_coordinates",
    "merchant_txn_count_5m",
    "merchant_distinct_accounts_5m",
)

CATEGORICAL_FEATURES = (
    "merchant_category",
    "location",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Selected feature set for the coverage-oriented Isolation Forest.
# Keep this in the same input order as the selected model artifact.
E_RATIO_CATEGORY_FEATURES = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "account_txn_count_60s",
    "account_txn_count_5m",
    "seconds_since_prev_account_txn",
    "has_prev_account_txn",
    "log_amount_deviation",
    "has_account_amount_history",
    "is_new_device_for_account",
    "location_changed",
    "merchant_txn_count_5m",
    "merchant_distinct_accounts_5m",
    "merchant_category",
    "location",
    "account_amount_ratio",
    "category_amount_deviation",
)

METADATA_COLUMNS = (
    "transaction_id",
    "timestamp",
    "account_id",
    "merchant_id",
    "fraud_type",
)

MIN_AMOUNT_HISTORY = 5
CATEGORY_AMOUNT_HISTORY_SIZE = 500
ROBUST_SCALE_FLOOR = 0.05
ROBUST_ZSCORE_CLIP = 20.0
IMPOSSIBLE_TRAVEL_SPEED_KMPH = 900.0

LOCATION_COORDINATES = {
    "US-CA": (36.7783, -119.4179),
    "US-NY": (42.9538, -75.5268),
    "US-TX": (31.0545, -99.9018),
    "GB-LON": (51.5074, -0.1278),
    "DE-BER": (52.5200, 13.4050),
    "IN-BLR": (12.9716, 77.5946),
    "SG-SIN": (1.3521, 103.8198),
}

EARTH_RADIUS_KM = 6371.0088


def _robust_zscore(value: float, history: list[float]) -> float:
    if len(history) < MIN_AMOUNT_HISTORY:
        return 0.0

    center = median(history)
    absolute_deviations = [abs(previous_value - center) for previous_value in history]
    mad = median(absolute_deviations)
    scale = 1.4826 * mad

    if scale < ROBUST_SCALE_FLOOR:
        q25, q75 = np.percentile(history, [25, 75])
        scale = (q75 - q25) / 1.349

    scale = max(float(scale), ROBUST_SCALE_FLOOR)
    score = (value - center) / scale

    return float(np.clip(score, -ROBUST_ZSCORE_CLIP, ROBUST_ZSCORE_CLIP))


def _haversine_km(
    coordinate_a: tuple[float, float],
    coordinate_b: tuple[float, float],
) -> float:
    lat1, lon1 = map(radians, coordinate_a)
    lat2, lon2 = map(radians, coordinate_b)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = sin(delta_lat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2.0) ** 2
    a = min(1.0, max(0.0, a))

    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a))


def _location_coordinates(location: str) -> tuple[float, float] | None:
    return LOCATION_COORDINATES.get(location.strip().upper())


def _to_record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)

    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if not isinstance(result, Mapping):
            raise TypeError("Transaction.to_dict() must return a mapping.")
        return dict(result)

    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)

    raise TypeError(
        "Each record must be a mapping, a dataclass instance, "
        "or an object exposing to_dict()."
    )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Transaction timestamp cannot be empty.")

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid transaction timestamp: {value!r}") from exc
    else:
        raise ValueError("Transaction timestamp must be a datetime or ISO-8601 string.")

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    else:
        result = result.astimezone(timezone.utc)

    return result


def _required_id(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Transaction field {field_name!r} must be a non-empty string."
        )
    return value.strip()


def _normalize_category(value: Any) -> str:
    if value is None:
        return MISSING_CATEGORY

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return MISSING_CATEGORY

    return text


def _parse_amount(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Transaction amount must be numeric, not boolean.")

    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid transaction amount: {value!r}") from exc

    if not isfinite(amount) or amount < 0:
        raise ValueError("Transaction amount must be finite and non-negative.")

    return amount


@dataclass
class _AccountHistory:
    recent_timestamps: deque[datetime] = field(default_factory=deque)
    amount_history: deque[float] = field(default_factory=deque)
    devices: set[str] = field(default_factory=set)
    last_timestamp: datetime | None = None
    last_location: str | None = None


@dataclass
class _MerchantHistory:
    recent_transactions: deque[tuple[datetime, str]] = field(default_factory=deque)


class HistoricalFeatureBuilder:
    """Build transaction features using only information from prior transactions."""

    def __init__(
        self,
        amount_history_size: int = 20,
        category_history_size: int = CATEGORY_AMOUNT_HISTORY_SIZE,
    ) -> None:
        for name, value in (
            ("amount_history_size", amount_history_size),
            ("category_history_size", category_history_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        self.amount_history_size = amount_history_size
        self.category_history_size = category_history_size
        self.reset()

    def reset(self) -> None:
        """Clear all account, merchant, and category history."""
        self._accounts: dict[str, _AccountHistory] = {}
        self._merchants: dict[str, _MerchantHistory] = {}
        self._category_amount_history: dict[str, deque[float]] = {}
        self._last_timestamp: datetime | None = None

    @staticmethod
    def _prune_account_history(
        history: _AccountHistory,
        cutoff: datetime,
    ) -> None:
        while history.recent_timestamps and history.recent_timestamps[0] < cutoff:
            history.recent_timestamps.popleft()

    @staticmethod
    def _prune_merchant_history(
        history: _MerchantHistory,
        cutoff: datetime,
    ) -> None:
        while (
            history.recent_transactions and history.recent_transactions[0][0] < cutoff
        ):
            history.recent_transactions.popleft()

    def transform_one(self, record: Any) -> dict[str, Any]:
        row = _to_record_dict(record)
        required_fields = (
            "account_id",
            "merchant_id",
            "device_id",
            "amount",
            "timestamp",
        )
        missing = [name for name in required_fields if name not in row]
        if missing:
            raise ValueError(f"Transaction is missing required fields: {missing}")

        account_id = _required_id(row, "account_id")
        merchant_id = _required_id(row, "merchant_id")
        device_id = _required_id(row, "device_id")
        timestamp = _parse_timestamp(row["timestamp"])
        amount = _parse_amount(row["amount"])
        merchant_category = _normalize_category(row.get("merchant_category"))
        location = _normalize_category(row.get("location"))

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                "Transactions must have strictly increasing timestamps. "
                f"Received {timestamp.isoformat()} after "
                f"{self._last_timestamp.isoformat()}."
            )

        account = self._accounts.setdefault(account_id, _AccountHistory())
        merchant = self._merchants.setdefault(merchant_id, _MerchantHistory())

        cutoff_5m = timestamp - timedelta(minutes=5)
        cutoff_60s = timestamp - timedelta(seconds=60)

        self._prune_account_history(account, cutoff_5m)
        self._prune_merchant_history(merchant, cutoff_5m)

        log_amount = log1p(amount)
        previous_transaction_exists = account.last_timestamp is not None

        if previous_transaction_exists:
            seconds_since_previous = (
                timestamp - account.last_timestamp
            ).total_seconds()
        else:
            seconds_since_previous = 0.0

        hour_fraction = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
            + timestamp.microsecond / 3_600_000_000.0
        ) / 24.0
        hour_angle = 2.0 * pi * hour_fraction

        weekday_fraction = (timestamp.weekday() + hour_fraction) / 7.0
        weekday_angle = 2.0 * pi * weekday_fraction

        account_count_60s = sum(
            1 for prior_time in account.recent_timestamps if prior_time >= cutoff_60s
        )
        account_count_5m = len(account.recent_timestamps)

        prior_amounts = list(account.amount_history)
        prior_log_amounts = [
            log1p(previous_amount) for previous_amount in prior_amounts
        ]
        has_amount_history = int(bool(prior_amounts))

        if has_amount_history:
            amount_deviation = log_amount - median(prior_log_amounts)
        else:
            amount_deviation = 0.0

        has_reliable_amount_history = len(prior_amounts) >= MIN_AMOUNT_HISTORY

        if has_reliable_amount_history:
            historical_median_amount = median(prior_amounts)
            if historical_median_amount > 0:
                account_amount_ratio = amount / historical_median_amount
            else:
                account_amount_ratio = 1.0 if amount == 0 else 100.0

            account_amount_ratio = min(account_amount_ratio, 100.0)
            account_amount_robust_zscore = _robust_zscore(
                log_amount,
                prior_log_amounts,
            )
        else:
            account_amount_ratio = 1.0
            account_amount_robust_zscore = 0.0

        category_history = self._category_amount_history.get(merchant_category)
        has_category_amount_history = int(
            category_history is not None and len(category_history) >= MIN_AMOUNT_HISTORY
        )

        if has_category_amount_history:
            category_amount_deviation = log_amount - median(category_history)
        else:
            category_amount_deviation = 0.0

        is_new_device = int(device_id not in account.devices)
        location_changed = int(
            account.last_location is not None and location != account.last_location
        )

        geo_distance_km = 0.0
        geo_travel_speed_kmph = 0.0
        impossible_travel_flag = 0
        has_geo_coordinates = 0

        if previous_transaction_exists:
            previous_coordinates = _location_coordinates(account.last_location)
            current_coordinates = _location_coordinates(location)

            if previous_coordinates is not None and current_coordinates is not None:
                geo_distance_km = _haversine_km(
                    previous_coordinates,
                    current_coordinates,
                )
                has_geo_coordinates = 1

                elapsed_hours = seconds_since_previous / 3600.0
                if elapsed_hours > 0:
                    raw_travel_speed = geo_distance_km / elapsed_hours
                    geo_travel_speed_kmph = min(raw_travel_speed, 10_000.0)
                    impossible_travel_flag = int(
                        raw_travel_speed > IMPOSSIBLE_TRAVEL_SPEED_KMPH
                    )

        merchant_accounts_5m = {
            prior_account_id for _, prior_account_id in merchant.recent_transactions
        }
        merchant_count_5m = len(merchant.recent_transactions)
        merchant_distinct_accounts_5m = len(merchant_accounts_5m)

        features: dict[str, Any] = {
            "log_amount": log_amount,
            "hour_sin": sin(hour_angle),
            "hour_cos": cos(hour_angle),
            "weekday_sin": sin(weekday_angle),
            "weekday_cos": cos(weekday_angle),
            "account_txn_count_60s": account_count_60s,
            "account_txn_count_5m": account_count_5m,
            "seconds_since_prev_account_txn": seconds_since_previous,
            "has_prev_account_txn": int(previous_transaction_exists),
            "log_amount_deviation": amount_deviation,
            "has_account_amount_history": has_amount_history,
            "account_amount_ratio": account_amount_ratio,
            "account_amount_robust_zscore": account_amount_robust_zscore,
            "has_reliable_account_amount_history": int(has_reliable_amount_history),
            "category_amount_deviation": category_amount_deviation,
            "has_category_amount_history": has_category_amount_history,
            "is_new_device_for_account": is_new_device,
            "location_changed": location_changed,
            "geo_distance_km": geo_distance_km,
            "geo_travel_speed_kmph": geo_travel_speed_kmph,
            "impossible_travel_flag": impossible_travel_flag,
            "has_geo_coordinates": has_geo_coordinates,
            "merchant_txn_count_5m": merchant_count_5m,
            "merchant_distinct_accounts_5m": merchant_distinct_accounts_5m,
            "merchant_category": merchant_category,
            "location": location,
        }

        # Update histories only after the current transaction's features
        # have been calculated, preventing current-row leakage.
        account.recent_timestamps.append(timestamp)
        account.amount_history.append(amount)

        while len(account.amount_history) > self.amount_history_size:
            account.amount_history.popleft()

        account.devices.add(device_id)
        account.last_timestamp = timestamp
        account.last_location = location

        merchant.recent_transactions.append((timestamp, account_id))

        category_history = self._category_amount_history.setdefault(
            merchant_category,
            deque(maxlen=self.category_history_size),
        )
        category_history.append(log_amount)

        self._last_timestamp = timestamp

        return features

    def transform(self, records: Iterable[Any]) -> pd.DataFrame:
        feature_rows = [self.transform_one(record) for record in records]
        return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)


def build_features(records: Iterable[Any]) -> pd.DataFrame:
    """Build all raw features for a chronological transaction sequence."""
    builder = HistoricalFeatureBuilder()
    return builder.transform(records)


def extract_labels(records: Iterable[Any]) -> np.ndarray:
    labels: list[int] = []

    for record in records:
        row = _to_record_dict(record)
        if "fraud_label" not in row:
            raise ValueError("Transaction is missing 'fraud_label'.")

        value = row["fraud_label"]

        if isinstance(value, str) and value.strip() in {"0", "1"}:
            label = int(value.strip())
        elif (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and value in (0, 1)
        ):
            label = int(value)
        elif isinstance(value, (bool, np.bool_)):
            label = int(value)
        else:
            raise ValueError(f"Invalid binary fraud label: {value!r}")

        labels.append(label)

    return np.asarray(labels, dtype=np.int8)


def extract_metadata(records: Iterable[Any]) -> pd.DataFrame:
    rows = [_to_record_dict(record) for record in records]

    return pd.DataFrame(
        [{column: row.get(column) for column in METADATA_COLUMNS} for row in rows],
        columns=METADATA_COLUMNS,
    )


def prepare_detection_dataset(
    records: Iterable[Any],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build raw features, labels, and metadata for one transaction split."""
    rows = [_to_record_dict(record) for record in records]

    builder = HistoricalFeatureBuilder()
    X = builder.transform(rows)
    y = extract_labels(rows)
    metadata = extract_metadata(rows)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("Feature, label, and metadata row counts do not agree.")

    return X, y, metadata


def _normalize_feature_columns(
    feature_columns: Iterable[str],
) -> tuple[str, ...]:
    """Validate and normalize an ordered set of raw feature names."""
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be an iterable of column names.")

    columns = tuple(feature_columns)

    if not columns:
        raise ValueError("At least one feature column is required.")

    if any(not isinstance(column, str) for column in columns):
        raise TypeError("Every feature column name must be a string.")

    if len(set(columns)) != len(columns):
        raise ValueError("Feature columns must not contain duplicates.")

    unknown = set(columns) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown feature columns: {sorted(unknown)}")

    return columns


def _validate_feature_frame(
    X: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Validate raw features and return the requested columns in order."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame of raw features.")

    if X.columns.duplicated().any():
        raise ValueError("Feature DataFrame contains duplicate columns.")

    columns = _normalize_feature_columns(feature_columns)

    missing_columns = set(columns) - set(X.columns)
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

    # Extra columns from the known raw feature schema are permitted because
    # a model may consume only a subset. Labels, identifiers, and any other
    # non-feature columns remain prohibited.
    unexpected_columns = set(X.columns) - set(FEATURE_COLUMNS)
    if unexpected_columns:
        raise ValueError(
            "Unexpected columns in model features: "
            f"{sorted(unexpected_columns)}. "
            "Keep labels and identifiers separate."
        )

    frame = X.loc[:, columns].copy()

    numeric_columns = [column for column in NUMERIC_FEATURES if column in columns]
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in columns
    ]

    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column!r} must be numeric.") from exc

        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Feature {column!r} contains missing or non-finite values."
            )

    for column in categorical_columns:
        frame[column] = frame[column].map(_normalize_category)

    return frame


def create_preprocessor(
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> ColumnTransformer:
    """Create a preprocessor while preserving the requested feature order."""

    columns = _normalize_feature_columns(feature_columns)

    numeric_columns = [column for column in columns if column in NUMERIC_FEATURES]

    categorical_columns = [
        column for column in columns if column in CATEGORICAL_FEATURES
    ]

    transformers = []

    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))

    if categorical_columns:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        transformers.append(("categorical", encoder, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    X_train: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> ColumnTransformer:
    """Fit a preprocessor using only the selected training features."""
    columns = _normalize_feature_columns(feature_columns)
    frame = _validate_feature_frame(
        X_train,
        feature_columns=columns,
    )

    if frame.empty:
        raise ValueError("Cannot fit preprocessing on an empty dataset.")

    preprocessor = create_preprocessor(columns)
    preprocessor.fit(frame)

    return preprocessor


def transform_features(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    feature_columns: Iterable[str] | None = None,
) -> np.ndarray:
    """Transform raw features with a fitted preprocessor.

    If feature_columns is omitted, the input schema is inferred from the
    fitted preprocessor's feature_names_in_ attribute.
    """
    if feature_columns is None:
        fitted_columns = getattr(preprocessor, "feature_names_in_", None)
        if fitted_columns is None:
            raise ValueError(
                "Cannot infer input feature columns from this preprocessor. "
                "Pass feature_columns explicitly."
            )
        feature_columns = tuple(str(name) for name in fitted_columns)

    columns = _normalize_feature_columns(feature_columns)
    frame = _validate_feature_frame(
        X,
        feature_columns=columns,
    )

    fitted_columns = getattr(preprocessor, "feature_names_in_", None)
    if fitted_columns is not None:
        expected_columns = tuple(str(name) for name in fitted_columns)
        if columns != expected_columns:
            raise ValueError(
                "Requested feature columns do not match the fitted "
                f"preprocessor schema. Expected {expected_columns}, got {columns}."
            )

    transformed = preprocessor.transform(frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    matrix = np.asarray(transformed, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Preprocessed features must form a 2-D matrix.")

    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed features contain missing or non-finite values.")

    return matrix
