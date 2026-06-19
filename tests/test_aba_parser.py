import pytest
from spiraton.data.aba_parser import ABAParser

def test_aba_parser_valid_line() -> None:
    line = "<SEG_A> <ADD><DX><OUT><ALPHA> Bonjour monde </SEG_A> <SEG_B> <ADD><DX><OUT><OMEGA> Soleil </SEG_B> <SEG_A_PRIME> <ADD><LV><IN><A_PRIME> Aurevoir <EOL> </SEG_A_PRIME> <EOL>"
    
    res = ABAParser.parse_line(line)
    assert res is not None
    assert res["op"] == "ADD"
    assert len(res["segments"]) == 3
    
    seg_a = res["segments"]["A"]
    assert seg_a["text"] == "Bonjour monde"
    assert seg_a["chiralite"] == "DX"
    assert seg_a["direction"] == "OUT"
    assert seg_a["position"] == "ALPHA"
    
    seg_a_prime = res["segments"]["A_PRIME"]
    assert seg_a_prime["text"] == "Aurevoir"
    assert seg_a_prime["chiralite"] == "LV"
    assert seg_a_prime["direction"] == "IN"
    assert seg_a_prime["position"] == "A_PRIME"

def test_aba_parser_invalid_line() -> None:
    line = "<SEG_A> Incomplet </SEG_A>"
    res = ABAParser.parse_line(line)
    assert res is None
