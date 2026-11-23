import pytest
import torch

from ecogdecode.data import SubjectData
from ecogdecode.synthetic import make_synthetic_subject

# PyTorch's default (every logical core) makes the many small operations of these tests very slow on hybrid CPUs
torch.set_num_threads(4)


@pytest.fixture(scope="session")
def synthetic_subject() -> SubjectData:
    return make_synthetic_subject()
