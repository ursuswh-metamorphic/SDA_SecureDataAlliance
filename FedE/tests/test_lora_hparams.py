import pytest

from lora_hparams import resolve_lora_hparams


def test_defaults_match_locked_b3_recipe():
    params = resolve_lora_hparams({})
    assert params == {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 1e-5,
    }


def test_rank_controls_default_alpha_and_lr_is_tunable():
    params = resolve_lora_hparams({
        "LORA_R": "32",
        "LEARNING_RATE": "0.0002",
    })
    assert params["rank"] == 32
    assert params["alpha"] == 64
    assert params["learning_rate"] == pytest.approx(2e-4)


def test_explicit_alpha_and_dropout_are_preserved():
    params = resolve_lora_hparams({
        "LORA_R": "16",
        "LORA_ALPHA": "16",
        "LORA_DROPOUT": "0.1",
    })
    assert params["alpha"] == 16
    assert params["dropout"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "env",
    [
        {"LORA_R": "0"},
        {"LORA_ALPHA": "0"},
        {"LORA_DROPOUT": "1"},
        {"LEARNING_RATE": "0"},
    ],
)
def test_invalid_values_fail_fast(env):
    with pytest.raises(ValueError):
        resolve_lora_hparams(env)
