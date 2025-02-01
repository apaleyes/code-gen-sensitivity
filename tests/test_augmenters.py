import pytest

from augmenters import get_augmenter

@pytest.mark.parametrize(
    "augmenter_name", ["keyboard", "synonym"]
)
def test_get_augmenter(augmenter_name):
    text = "some random english text to be augmented"
    rate = 0.9 # very high rate to ensure text is altered

    augmenter = get_augmenter(augmenter_name, rate)
    new_text = augmenter.augment(text)

    assert augmenter.name.lower() == augmenter_name
    assert augmenter.rate == rate
    assert text != new_text
