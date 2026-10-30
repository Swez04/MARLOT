import numpy as np
import pytest

from detection.evaluation import (
    evaluate_by_fraud_type,
    evaluate_scores,
    select_f1_threshold,
)
from detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)


@pytest.fixture
def training_features():
    rng = np.random.default_rng(7)
    return rng.normal(0, 0.3, size=(250, 4)).astype(np.float32)


def test_fit_and_score(training_features):
    detector = IsolationForestDetector().fit(training_features)

    scores = detector.score_samples(training_features)

    assert scores.shape == (250,)
    assert np.isfinite(scores).all()


def test_score_ordering_is_higher_for_anomalies(training_features):
    detector = IsolationForestDetector().fit(training_features)

    normal = np.zeros((1, 4), dtype=np.float32)
    unusual = np.full((1, 4), 8.0, dtype=np.float32)

    normal_score = detector.score_samples(normal)[0]
    unusual_score = detector.score_samples(unusual)[0]

    assert unusual_score > normal_score


def test_predict_requires_threshold(training_features):
    detector = IsolationForestDetector().fit(training_features)

    with pytest.raises(RuntimeError, match="threshold"):
        detector.predict(training_features[:2])


def test_threshold_and_prediction(training_features):
    detector = IsolationForestDetector().fit(training_features)

    threshold = float(np.median(detector.score_samples(training_features)))
    detector.set_threshold(threshold)

    predictions = detector.predict(training_features)

    assert predictions.shape == (250,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_invalid_training_data():
    detector = IsolationForestDetector()

    with pytest.raises(ValueError, match="non-empty"):
        detector.fit(np.empty((0, 4)))

    with pytest.raises(ValueError, match="NaN or infinite"):
        detector.fit(np.array([[0.0, np.nan], [1.0, 2.0]]))


def test_invalid_configuration():
    with pytest.raises(ValueError, match="n_estimators"):
        IsolationForestConfig(n_estimators=0)

    with pytest.raises(ValueError, match="contamination"):
        IsolationForestConfig(contamination=0.8)


def test_threshold_selection_uses_validation_labels():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold, diagnostics = select_f1_threshold(y, scores)

    predictions = (scores >= threshold).astype(int)

    assert diagnostics["validation_f1"] == pytest.approx(1.0)
    assert predictions.tolist() == y.tolist()


def test_metrics_match_known_example():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = evaluate_scores(y, scores, threshold=0.75)

    assert metrics["confusion_matrix"] == {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_per_fraud_type_recall():
    y = np.array([0, 1, 1, 1])
    fraud_types = np.array(["", "velocity", "velocity", "collusion"])
    predictions = np.array([0, 1, 0, 1])

    metrics = evaluate_by_fraud_type(y, fraud_types, predictions)

    assert metrics["velocity"]["support"] == 2
    assert metrics["velocity"]["recall"] == pytest.approx(0.5)
    assert metrics["collusion"]["support"] == 1
    assert metrics["collusion"]["recall"] == pytest.approx(1.0)


def test_save_reload_preserves_scores_and_threshold(
    training_features,
    tmp_path,
):
    detector = IsolationForestDetector().fit(training_features)
    detector.set_threshold(0.55)

    expected = detector.score_samples(training_features[:10])
    artifact_path = tmp_path / "isolation_forest.joblib"

    detector.save(artifact_path)
    loaded = IsolationForestDetector.load(artifact_path)

    actual = loaded.score_samples(training_features[:10])

    np.testing.assert_allclose(actual, expected)
    assert loaded.threshold == pytest.approx(0.55)
    np.testing.assert_array_equal(
        loaded.predict(training_features[:10]),
        (expected >= 0.55).astype(np.int8),
    )
