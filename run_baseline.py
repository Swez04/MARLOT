"""
Phase 1 + 2 entry point:
  1. Generate synthetic transactions (train set: normal-ish, test set: mixed fraud)
  2. Fit each baseline detector
  3. Print a metrics comparison table

Run: python run_baseline.py
"""

from simulator.generator import generate_stream, generate_regime_sequence
from detectors.rules import RuleBasedDetector
from detectors.isolation_forest import IsolationForestDetector
from detectors.autoencoder import AutoencoderDetector
from evaluation.metrics import evaluate, time_detector


def main():
    print("Generating data...")
    train_txns = generate_stream(2000, regime="normal", seed=1)
    test_txns = generate_regime_sequence(
        ["normal", "amount_spike", "velocity", "geo_hop", "collusion"],
        n_per_regime=400,
        seed=2,
    )
    y_true = [t.fraud_label for t in test_txns]

    detectors = {
        "RuleBased": RuleBasedDetector(),
        "IsolationForest": IsolationForestDetector(),
        "Autoencoder": AutoencoderDetector(),
    }

    print(f"\n{'Detector':<18}{'Precision':<12}{'Recall':<12}{'F1':<12}{'FP Rate':<12}{'TPS':<10}")
    print("-" * 76)

    for name, detector in detectors.items():
        if hasattr(detector, "fit"):
            detector.fit(train_txns)

        timing = time_detector(detector, test_txns, fit_first=False)
        y_pred = timing["predictions"]
        m = evaluate(y_true, y_pred)

        print(f"{name:<18}{m['precision']:<12.3f}{m['recall']:<12.3f}"
              f"{m['f1']:<12.3f}{m['false_positive_rate']:<12.3f}{timing['throughput_tps']:<10.0f}")


if __name__ == "__main__":
    main()
