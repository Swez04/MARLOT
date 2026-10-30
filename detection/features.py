"""Leakage-safe feature engineering for MARLOT transaction detection."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from math import (
    asin,
    cos,
    isfinite,
    log1p,
    pi,
    radians,
    sin,
    sqrt,
)
from statistics import median
from typing import Any
from pathlib import Path

from scipy import sparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_CATEGORY = "__MISSING__"

# Features available to the models, in a deterministic order.
NUMERIC_FEATURES = (
    # Transaction-level
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    # Account activity
    "account_txn_count_60s",
    "account_txn_count_5m",
    "seconds_since_prev_account_txn",
    "has_prev_account_txn",
    # Account spending behaviour
    "log_amount_deviation",
    "has_account_amount_history",
    "account_amount_ratio",
    "account_amount_robust_zscore",
    "has_reliable_account_amount_history",
    # Merchant-category spending behaviour
    "category_amount_deviation",
    "has_category_amount_history",
    # Account and device behaviour
    "is_new_device_for_account",
    "location_changed",
    # Geographical behaviour
    "geo_distance_km",
    "geo_travel_speed_kmph",
    "impossible_travel_flag",
    "has_geo_coordinates",
    # Merchant behaviour
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


# Minimum historical observations needed before behavioural statistics
# are considered reliable.
MIN_AMOUNT_HISTORY = 5

# Keep only the latest N observations for each merchant category.
CATEGORY_AMOUNT_HISTORY_SIZE = 500

# Numerical safeguards for robust Z-scores.
ROBUST_SCALE_FLOOR = 0.05
ROBUST_ZSCORE_CLIP = 20.0

# Approximate threshold for suspiciously fast geographical movement.
IMPOSSIBLE_TRAVEL_SPEED_KMPH = 900.0

# Approximate representative coordinates for simulator location codes.
# State-level locations use approximate regional centres.
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


def _robust_zscore(
    value: float,
    history: list[float],
) -> float:
    """
    Calculate a robust Z-score using historical median and MAD.

    The calculation operates on log-transformed amounts.

    Falls back to IQR-based scaling when MAD is zero or very small.
    Clips extreme values to avoid uncontrolled numerical magnitudes.
    """

    if len(history) < MIN_AMOUNT_HISTORY:
        return 0.0

    center = median(history)

    absolute_deviations = [abs(previous_value - center) for previous_value in history]

    mad = median(absolute_deviations)

    # 1.4826 makes MAD approximately comparable to standard deviation
    # for normally distributed observations.
    scale = 1.4826 * mad

    if scale < ROBUST_SCALE_FLOOR:
        q25, q75 = np.percentile(history, [25, 75])
        scale = (q75 - q25) / 1.349

    scale = max(float(scale), ROBUST_SCALE_FLOOR)

    score = (value - center) / scale

    return float(
        np.clip(
            score,
            -ROBUST_ZSCORE_CLIP,
            ROBUST_ZSCORE_CLIP,
        )
    )


def _haversine_km(
    coordinate_a: tuple[float, float],
    coordinate_b: tuple[float, float],
) -> float:
    """
    Calculate the approximate great-circle distance between two
    latitude/longitude coordinates.
    """

    lat1, lon1 = map(radians, coordinate_a)
    lat2, lon2 = map(radians, coordinate_b)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = sin(delta_lat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2.0) ** 2

    a = min(1.0, max(0.0, a))

    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a))


def _location_coordinates(
    location: str,
) -> tuple[float, float] | None:
    """Look up approximate coordinates for a simulator location."""

    return LOCATION_COORDINATES.get(location.strip().upper())


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

    recent_timestamps: deque[datetime] = field(default_factory=deque)

    # Raw amounts from the latest N account transactions.
    amount_history: deque[float] = field(default_factory=deque)

    devices: set[str] = field(default_factory=set)

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
        """Discard all accumulated feature history."""

        self._accounts: dict[str, _AccountHistory] = {}
        self._merchants: dict[str, _MerchantHistory] = {}

        # Historical log amounts grouped by merchant category.
        self._category_amount_history: dict[str, deque[float]] = {}

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
        Build features for one transaction using historical information only.

        Features are calculated before updating account, merchant, and
        category histories. This prevents the current transaction from
        influencing its own historical baseline.
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

        # All transactions must arrive in strictly chronological order.
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                "Transactions must have strictly increasing timestamps. "
                f"Received {timestamp.isoformat()} after "
                f"{self._last_timestamp.isoformat()}."
            )

        account = self._accounts.setdefault(account_id, _AccountHistory())
        merchant = self._merchants.setdefault(merchant_id, _MerchantHistory())

        # ---------------------------------------------------------
        # 1. Prune rolling historical windows
        # ---------------------------------------------------------

        cutoff_5m = timestamp - timedelta(minutes=5)
        cutoff_60s = timestamp - timedelta(seconds=60)

        self._prune_account_history(account, cutoff_5m)
        self._prune_merchant_history(merchant, cutoff_5m)

        # ---------------------------------------------------------
        # 2. Transaction-level features
        # ---------------------------------------------------------

        log_amount = log1p(amount)

        # Time since the account's previous transaction.
        previous_transaction_exists = account.last_timestamp is not None

        if previous_transaction_exists:
            seconds_since_previous = (
                timestamp - account.last_timestamp
            ).total_seconds()
        else:
            seconds_since_previous = 0.0

        # Cyclical time-of-day features.
        hour_fraction = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
            + timestamp.microsecond / 3_600_000_000.0
        ) / 24.0

        hour_angle = 2.0 * pi * hour_fraction

        # Weekly cycle includes the time of day.
        weekday_fraction = (timestamp.weekday() + hour_fraction) / 7.0

        weekday_angle = 2.0 * pi * weekday_fraction

        # ---------------------------------------------------------
        # 3. Account activity features
        # ---------------------------------------------------------

        account_count_60s = sum(
            1 for prior_time in account.recent_timestamps if prior_time >= cutoff_60s
        )

        account_count_5m = len(account.recent_timestamps)

        # ---------------------------------------------------------
        # 4. Account-specific spending features
        # ---------------------------------------------------------

        # amount_history contains raw amounts, not log amounts.
        prior_amounts = list(account.amount_history)

        # Convert historical amounts to log scale for deviation analysis.
        prior_log_amounts = [
            log1p(previous_amount) for previous_amount in prior_amounts
        ]

        has_amount_history = int(bool(prior_amounts))

        if has_amount_history:
            amount_deviation = log_amount - median(prior_log_amounts)
        else:
            amount_deviation = 0.0

        # Require enough observations for reliable spending statistics.
        has_reliable_amount_history = len(prior_amounts) >= MIN_AMOUNT_HISTORY

        if has_reliable_amount_history:
            historical_median_amount = median(prior_amounts)

            if historical_median_amount > 0:
                account_amount_ratio = amount / historical_median_amount
            else:
                # Handle zero-valued historical transactions safely.
                account_amount_ratio = 1.0 if amount == 0 else 100.0

            # Prevent extreme ratios from dominating the feature.
            account_amount_ratio = min(
                account_amount_ratio,
                100.0,
            )

            account_amount_robust_zscore = _robust_zscore(
                log_amount,
                prior_log_amounts,
            )

        else:
            # Neutral values until enough historical observations exist.
            account_amount_ratio = 1.0
            account_amount_robust_zscore = 0.0

        # ---------------------------------------------------------
        # 5. Merchant-category spending features
        # ---------------------------------------------------------

        category_history = self._category_amount_history.get(merchant_category)

        has_category_amount_history = int(
            category_history is not None and len(category_history) >= MIN_AMOUNT_HISTORY
        )

        if has_category_amount_history:
            category_amount_deviation = log_amount - median(category_history)
        else:
            category_amount_deviation = 0.0

        # ---------------------------------------------------------
        # 6. Account and device features
        # ---------------------------------------------------------

        is_new_device = int(device_id not in account.devices)

        location_changed = int(
            account.last_location is not None and location != account.last_location
        )

        # ---------------------------------------------------------
        # 7. Geographical features
        # ---------------------------------------------------------

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

                    geo_travel_speed_kmph = min(
                        raw_travel_speed,
                        10_000.0,
                    )

                    impossible_travel_flag = int(
                        raw_travel_speed > IMPOSSIBLE_TRAVEL_SPEED_KMPH
                    )

        # ---------------------------------------------------------
        # 8. Merchant activity features
        # ---------------------------------------------------------

        merchant_accounts_5m = {
            prior_account_id for _, prior_account_id in merchant.recent_transactions
        }

        merchant_count_5m = len(merchant.recent_transactions)

        merchant_distinct_accounts_5m = len(merchant_accounts_5m)

        # ---------------------------------------------------------
        # 9. Assemble features
        # ---------------------------------------------------------

        features: dict[str, Any] = {
            # Transaction-level
            "log_amount": log_amount,
            "hour_sin": sin(hour_angle),
            "hour_cos": cos(hour_angle),
            "weekday_sin": sin(weekday_angle),
            "weekday_cos": cos(weekday_angle),
            # Account activity
            "account_txn_count_60s": account_count_60s,
            "account_txn_count_5m": account_count_5m,
            "seconds_since_prev_account_txn": seconds_since_previous,
            "has_prev_account_txn": int(previous_transaction_exists),
            # Account spending
            "log_amount_deviation": amount_deviation,
            "has_account_amount_history": has_amount_history,
            "account_amount_ratio": account_amount_ratio,
            "account_amount_robust_zscore": account_amount_robust_zscore,
            "has_reliable_account_amount_history": int(has_reliable_amount_history),
            # Merchant-category spending
            "category_amount_deviation": category_amount_deviation,
            "has_category_amount_history": has_category_amount_history,
            # Account and device
            "is_new_device_for_account": is_new_device,
            "location_changed": location_changed,
            # Geographical
            "geo_distance_km": geo_distance_km,
            "geo_travel_speed_kmph": geo_travel_speed_kmph,
            "impossible_travel_flag": impossible_travel_flag,
            "has_geo_coordinates": has_geo_coordinates,
            # Merchant activity
            "merchant_txn_count_5m": merchant_count_5m,
            "merchant_distinct_accounts_5m": merchant_distinct_accounts_5m,
            # Categorical
            "merchant_category": merchant_category,
            "location": location,
        }

        # ---------------------------------------------------------
        # 10. Update historical state AFTER feature extraction
        # ---------------------------------------------------------

        # Account rolling transaction history.
        account.recent_timestamps.append(timestamp)

        # Store raw amounts for account-specific spending statistics.
        account.amount_history.append(amount)

        while len(account.amount_history) > self.amount_history_size:
            account.amount_history.popleft()

        # Account devices and previous transaction information.
        account.devices.add(device_id)
        account.last_timestamp = timestamp
        account.last_location = location

        # Merchant rolling history.
        merchant.recent_transactions.append((timestamp, account_id))

        # Category history stores log-transformed amounts.
        # The deque automatically discards the oldest observation
        # when its configured maximum length is reached.
        category_history = self._category_amount_history.setdefault(
            merchant_category,
            deque(maxlen=self.category_history_size),
        )

        category_history.append(log_amount)

        # Maintain global chronological ordering.
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


def save_processed_split(
    path: str | Path,
    X: Any,
    y: Any,
    transaction_ids: Any,
    fraud_types: Any,
) -> None:
    """Save a processed feature split without losing row alignment."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y, dtype=np.int8)
    transaction_ids = np.asarray(transaction_ids, dtype=str)
    fraud_types = np.asarray(fraud_types, dtype=str)

    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X = np.asarray(X, dtype=np.float32)

    n = X.shape[0]

    if X.ndim != 2 or n == 0:
        raise ValueError("X must be a non-empty 2D feature matrix")

    if not (len(y) == n and len(transaction_ids) == n and len(fraud_types) == n):
        raise ValueError("Features, labels, IDs, and fraud types must align")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")

    if len(set(transaction_ids.tolist())) != n:
        raise ValueError("Transaction IDs must be unique within the split")

    if np.any((y == 0) & (fraud_types != "")):
        raise ValueError("Legitimate records must have empty fraud_type")

    if np.any((y == 1) & (fraud_types == "")):
        raise ValueError("Fraudulent records must have a fraud_type")

    np.savez_compressed(
        path,
        X=X,
        y=y,
        transaction_id=transaction_ids,
        fraud_type=fraud_types,
    )
