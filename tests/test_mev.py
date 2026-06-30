import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mev.sandwich import detect_sandwiches
from src.mev.zeromev_client import index_by_tx, label_for_tx, is_mev


def test_detects_classic_sandwich():
    # attacker BUY (idx1) → victim BUY (idx2) → attacker SELL (idx3), aynı token
    swaps = [
        {"address": "0xATT", "token": "0xTKN", "side": "BUY", "tx_index": 1, "est_value_usd": 50000},
        {"address": "0xVIC", "token": "0xTKN", "side": "BUY", "tx_index": 2, "est_value_usd": 8000},
        {"address": "0xATT", "token": "0xTKN", "side": "SELL", "tx_index": 3, "est_value_usd": 51000},
    ]
    res = detect_sandwiches(swaps)
    assert len(res) == 1
    s = res[0]
    assert s.attacker == "0xatt" and s.token == "0xtkn"
    assert s.front_idx == 1 and s.back_idx == 3
    assert s.victim_indices == [2] and s.victim_volume_usd == 8000


def test_no_sandwich_without_victim():
    swaps = [
        {"address": "0xATT", "token": "0xTKN", "side": "BUY", "tx_index": 1, "est_value_usd": 50000},
        {"address": "0xATT", "token": "0xTKN", "side": "SELL", "tx_index": 2, "est_value_usd": 51000},
    ]
    assert detect_sandwiches(swaps) == []


def test_zeromev_indexing_and_labels():
    rows = [
        {"block_number": 100, "tx_index": 5, "mev_type": "sandwich"},
        {"block_number": 100, "tx_index": 6, "mev_type": "swap"},
    ]
    idx = index_by_tx(rows)
    assert label_for_tx(idx, 5) == "sandwich"
    assert label_for_tx(idx, 99) is None
    assert is_mev("sandwich") is True
    assert is_mev("swap") is False
    assert is_mev(None) is False


if __name__ == "__main__":
    test_detects_classic_sandwich(); test_no_sandwich_without_victim()
    test_zeromev_indexing_and_labels()
    print("MEV testleri GEÇTİ")
