"""Smoke test a running transaction simulator API."""

import os
import sys

import httpx2


def main() -> int:
    base_url = os.getenv(
        "API_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")

    payload = {
        "scenario": "normal",
        "count": 10,
        "seed": 42,
        "start_time": "2026-10-20T10:00:00",
        "include_transactions": True,
    }

    try:
        with httpx2.Client(
            base_url=base_url,
            timeout=10.0,
        ) as client:

            # 1. Check process health.
            health = client.get("/health")
            health.raise_for_status()

            if health.json() != {"status": "ok"}:
                raise RuntimeError("Unexpected health response.")

            print("[PASS] Health endpoint")

            # 2. Check that the OpenAPI document is served.
            docs = client.get("/openapi.json")
            docs.raise_for_status()

            paths = docs.json()["paths"]

            if "/simulate" not in paths:
                raise RuntimeError("Simulation endpoint missing from OpenAPI.")

            print("[PASS] OpenAPI document")

            # 3. Generate a small transaction batch.
            response = client.post(
                "/simulate",
                json=payload,
            )
            response.raise_for_status()

            body = response.json()

            if body["generated_count"] != payload["count"]:
                raise RuntimeError("Generated count does not match requested count.")

            transactions = body["transactions"]

            if len(transactions) != payload["count"]:
                raise RuntimeError("Transaction payload has an incorrect length.")

            # 4. Verify transaction IDs are unique within the batch.
            transaction_ids = [txn["transaction_id"] for txn in transactions]

            if len(transaction_ids) != len(set(transaction_ids)):
                raise RuntimeError("Duplicate transaction IDs within the batch.")

            # 5. Check that summary counts match the transactions.
            summary = body["summary"]

            if (
                summary["legitimate_count"] + summary["fraudulent_count"]
                != body["generated_count"]
            ):
                raise RuntimeError("Summary counts do not match generated count.")

            print("[PASS] Simulation generation")
            print("[PASS] Transaction ID uniqueness")
            print("[PASS] Summary consistency")

            # 6. Verify invalid input is rejected.
            invalid_payload = {
                **payload,
                "count": -1,
            }

            invalid_response = client.post(
                "/simulate",
                json=invalid_payload,
            )

            if invalid_response.status_code != 422:
                raise RuntimeError("Negative count was not rejected with HTTP 422.")

            print("[PASS] Request validation")

    except (httpx2.HTTPError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[FAIL] Smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
