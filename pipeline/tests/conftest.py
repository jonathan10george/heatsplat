import pytest

from heatsplat.synth import SynthResult, build_house


@pytest.fixture(scope="session")
def house() -> SynthResult:
    return build_house()
