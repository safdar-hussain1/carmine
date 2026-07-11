import cv2
import numpy as np
import pytest
from skimage import data

from virtual_makeup.landmarks import FaceLandmarker


@pytest.fixture(scope="session")
def astronaut_bgr() -> np.ndarray:
    """Public-domain NASA portrait (Eileen Collins) bundled with skimage."""
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="session")
def landmarker() -> FaceLandmarker:
    lm = FaceLandmarker()
    yield lm
    lm.close()


@pytest.fixture(scope="session")
def astronaut_landmarks(landmarker, astronaut_bgr) -> np.ndarray:
    return landmarker.detect(astronaut_bgr)
