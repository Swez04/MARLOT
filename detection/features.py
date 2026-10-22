"""Leakage-safe feature engineering for MARLOT transaction detection."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from math import cos, isfinite, log1p, pi, sin
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_CATEGORY = "__MISSING__"

# Features available to the models, in a deterministic order.
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
    "is_new_device_for_account",
    "location_changed",
    "merchant_txn_count_5m",
    "merchant_distinct_accounts_5m",
)

CATEGORICAL_FEATURES = (
    "merchant_category",
    "location",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# These columns must never enter the model feature matrix.
FORBIDDEN_FEATURE_COLUMNS = {
    "transaction_id",
    "account_id",
    "merchant_id",
    "device_id",
    "fraud_label",
    "fraud_type",
}

METADATA_COLUMNS = (
    "transaction_id",
    "timestamp",
    "account_id",
    "merchant_id",
    "fraud_type",
)


def _to_record_dict(record: Any) -> dict[str, Any]:
    """Convert a transaction object or mapping into a dictionary."""

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
    """Parse a timestamp and normalize it to timezone-aware UTC."""

    if isinstance(value, datetime):
        result = value

    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Transaction timestamp cannot be empty.")

        # datetime.fromisoformat() supports offsets. Normalize the
        # common JSON ISO-8601 UTC suffix as well.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid transaction timestamp: {value!r}") from exc

    else:
        raise ValueError("Transaction timestamp must be a datetime or ISO-8601 string.")

    # Explicit convention: timestamps without a timezone are interpreted
    # as UTC. The simulator should ideally generate consistent timestamps.
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    else:
        result = result.astimezone(timezone.utc)

    return result


def _required_id(record: Mapping[str, Any], field_name: str) -> str:
    """Read a required non-empty entity identifier."""

    value = record.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Transaction field {field_name!r} must be a non-empty string."
        )

    return value.strip()


def _normalize_category(value: Any) -> str:
    """Represent missing or blank categorical values consistently."""

    if value is None:
        return MISSING_CATEGORY

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return MISSING_CATEGORY

    return text


def _parse_amount(value: Any) -> float:
    """Parse and validate a transaction amount."""

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
    """Private historical state used exclusively for feature extraction."""

    # Timestamps needed for rolling account transaction counts.
    recent_timestamps: deque[datetime] = field(default_factory=deque)

    # Log amounts from the last N transactions, regardless of elapsed time.
    amount_history: deque[float] = field(default_factory=deque)

    # All previously observed devices for this account.
    devices: set[str] = field(default_factory=set)

    # Retained even when a transaction is older than the rolling window.
    last_timestamp: datetime | None = None
    last_location: str | None = None


@dataclass
class _MerchantHistory:
    """Private rolling history for a merchant."""

    # Each entry contains (timestamp, account_id).
    recent_transactions: deque[tuple[datetime, str]] = field(default_factory=deque)


class HistoricalFeatureBuilder:
    """
    Construct behavioral features using only transactions observed earlier.

    Records must arrive in strictly increasing timestamp order. For every
    record, features are calculated first; history is updated afterward.

    A builder is stateful. Use reset() between independent datasets, or
    create a new builder for each dataset.
    """

    def __init__(self, amount_history_size: int = 20) -> None:
        if (
            isinstance(amount_history_size, bool)
            or not isinstance(amount_history_size, int)
            or amount_history_size < 1
        ):
            raise ValueError("amount_history_size must be a positive integer.")

        self.amount_history_size = amount_history_size
        self.reset()

    def reset(self) -> None:
        """Discard all accumulated feature history."""

        self._accounts: dict[str, _AccountHistory] = {}
        self._merchants: dict[str, _MerchantHistory] = {}
        self._last_timestamp: datetime | None = None

    @staticmethod
    def _prune_account_history(
        history: _AccountHistory,
        cutoff: datetime,
    ) -> None:
        """Retain timestamps at or after the rolling-window boundary."""

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
        """
        Build features for one transaction, then update historical state.

        The current transaction is excluded from all historical features.
        """

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

        # Historical windows use [timestamp - window, timestamp).
        # The current transaction is not yet part of either history.
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

        if account.amount_history:
            amount_deviation = log_amount - median(account.amount_history)
            has_amount_history = 1
        else:
            amount_deviation = 0.0
            has_amount_history = 0

        # Fractional hour in UTC, including minutes, seconds and microseconds.
        hour_fraction = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
            + timestamp.microsecond / 3_600_000_000.0
        ) / 24.0

        hour_angle = 2.0 * pi * hour_fraction

        # Include the time of day in the weekly cycle.
        weekday_fraction = (timestamp.weekday() + hour_fraction) / 7.0

        weekday_angle = 2.0 * pi * weekday_fraction

        account_count_60s = sum(
            1 for prior_time in account.recent_timestamps if prior_time >= cutoff_60s
        )

        merchant_accounts_5m = {
            prior_account_id for _, prior_account_id in merchant.recent_transactions
        }

        # Construct every feature before adding the current transaction
        # to account/merchant histories.
        features: dict[str, Any] = {
            "log_amount": log_amount,
            "hour_sin": sin(hour_angle),
            "hour_cos": cos(hour_angle),
            "weekday_sin": sin(weekday_angle),
            "weekday_cos": cos(weekday_angle),
            "account_txn_count_60s": account_count_60s,
            "account_txn_count_5m": len(account.recent_timestamps),
            "seconds_since_prev_account_txn": seconds_since_previous,
            "has_prev_account_txn": int(previous_transaction_exists),
            "log_amount_deviation": amount_deviation,
            "has_account_amount_history": has_amount_history,
            "is_new_device_for_account": int(device_id not in account.devices),
            "location_changed": int(
                account.last_location is not None and location != account.last_location
            ),
            "merchant_txn_count_5m": len(merchant.recent_transactions),
            "merchant_distinct_accounts_5m": len(merchant_accounts_5m),
            "merchant_category": merchant_category,
            "location": location,
        }

        # Update historical feature state exactly once after extracting
        # the current transaction's features.
        account.recent_timestamps.append(timestamp)
        account.amount_history.append(log_amount)

        while len(account.amount_history) > self.amount_history_size:
            account.amount_history.popleft()

        account.devices.add(device_id)
        account.last_timestamp = timestamp
        account.last_location = location

        merchant.recent_transactions.append((timestamp, account_id))

        self._last_timestamp = timestamp

        return features

    def transform(self, records: Iterable[Any]) -> pd.DataFrame:
        """
        Build features for an ordered sequence of records.

        This method preserves history across calls. Call reset() before
        processing a separate dataset.
        """

        feature_rows = [self.transform_one(record) for record in records]

        return pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLUMNS,
        )


def build_features(records: Iterable[Any]) -> pd.DataFrame:
    """Build features from one independent dataset with fresh history."""

    builder = HistoricalFeatureBuilder()
    return builder.transform(records)


def extract_labels(records: Iterable[Any]) -> np.ndarray:
    """Extract binary ground-truth labels separately from model features."""

    labels: list[int] = []

    for record in records:
        row = _to_record_dict(record)

        if "fraud_label" not in row:
            raise ValueError("Transaction is missing 'fraud_label'.")

        value = row["fraud_label"]

        # Be tolerant of JSONL files containing textual binary labels,
        # while rejecting ambiguous or invalid values.
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            label = int(value.strip())
        elif (
            not isinstance(value, bool)
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
    """Extract identifiers and fraud information for evaluation only."""

    rows = [_to_record_dict(record) for record in records]

    return pd.DataFrame(
        [{column: row.get(column) for column in METADATA_COLUMNS} for row in rows],
        columns=METADATA_COLUMNS,
    )


def prepare_detection_dataset(
    records: Iterable[Any],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Prepare an independent dataset.

    Returns:
        X: leakage-safe raw feature DataFrame.
        y: ground-truth labels, aligned with X.
        metadata: identifiers and evaluation information, also aligned.
    """

    rows = [_to_record_dict(record) for record in records]

    builder = HistoricalFeatureBuilder()
    X = builder.transform(rows)
    y = extract_labels(rows)
    metadata = extract_metadata(rows)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("Feature, label, and metadata row counts do not agree.")

    return X, y, metadata


def _validate_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a clean raw feature frame."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame of raw features.")

    if X.columns.duplicated().any():
        raise ValueError("Feature DataFrame contains duplicate columns.")

    missing_columns = set(FEATURE_COLUMNS) - set(X.columns)
    extra_columns = set(X.columns) - set(FEATURE_COLUMNS)

    if missing_columns:
        raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

    if extra_columns:
        raise ValueError(
            "Unexpected columns in model features: "
            f"{sorted(extra_columns)}. Keep labels and identifiers separate."
        )

    frame = X.loc[:, FEATURE_COLUMNS].copy()

    for column in NUMERIC_FEATURES:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column!r} must be numeric.") from exc

        values = frame[column].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Feature {column!r} contains missing or non-finite values."
            )

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].map(_normalize_category)

    return frame


def create_preprocessor() -> ColumnTransformer:
    """Create a new unfitted preprocessing pipeline."""

    # sparse_output is the current scikit-learn argument. The fallback
    # allows compatibility with older versions that use sparse instead.
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

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                encoder,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Fit preprocessing on training features only.

    Do not call this function on validation or test data.
    """

    frame = _validate_feature_frame(X_train)

    if frame.empty:
        raise ValueError("Cannot fit preprocessing on an empty dataset.")

    preprocessor = create_preprocessor()
    preprocessor.fit(frame)

    return preprocessor


def transform_features(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Transform a feature DataFrame using an already-fitted preprocessor."""

    frame = _validate_feature_frame(X)

    transformed = preprocessor.transform(frame)

    # Return a dense numerical array suitable for scikit-learn and PyTorch.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    matrix = np.asarray(transformed, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Preprocessed features must form a 2-D matrix.")

    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed features contain missing or non-finite values.")

    return matrix
