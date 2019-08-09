import nose
from psv.config import L, C


def test_psv_dataset():
    from psv.ptdataset import PsvDataset

    ds = PsvDataset()

    assert len(ds == 956), "The dataset should have this many entries"
    


