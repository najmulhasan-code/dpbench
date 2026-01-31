"""Test all model connections before running full experiments."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.models import create_model

TEST_SYSTEM = "You are a helpful assistant. Respond concisely."
TEST_USER = "Say 'Hello' and nothing else."


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_model(name: str, provider: str, model_id: str) -> bool:
    """Test a single model connection."""
    print(f"\nTesting {name}...")
    print(f"  Provider: {provider}")
    print(f"  Model ID: {model_id}")

    try:
        model_fn = create_model(
            provider=provider,
            model_id=model_id,
            temperature=0.0,
            max_tokens=1024,
        )
        response = model_fn(TEST_SYSTEM, TEST_USER)
        print(f"  Response: {response.text[:50]}...")
        print(f"  Status: OK")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Status: FAILED")
        return False


def main():
    """Test all configured models from models.yaml."""
    print("=" * 60)
    print("DPBench Model Connection Test")
    print("=" * 60)

    # Load models from YAML config
    configs_dir = Path(__file__).parent.parent / "configs"
    models_config = load_config(configs_dir / "models.yaml")

    results = []
    for model_name, model_cfg in models_config["models"].items():
        success = test_model(
            name=model_name,
            provider=model_cfg["provider"],
            model_id=model_cfg["model_id"],
        )
        results.append((model_name, success))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)

    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFix failed models before running experiments.")
        print("Check your .env file for missing API keys.")
        sys.exit(1)
    else:
        print("\nAll models ready for experiments!")
        sys.exit(0)


if __name__ == "__main__":
    main()
