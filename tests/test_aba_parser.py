import pytest

from spiraton.data.aba import (
    parse_aba_line,
    try_parse_aba_line,
    validate_cycle,
    is_terminator_line,
    AbaParseError,
)


# Lignes réelles tirées des corpus (fixtures figées dans le test).
LINE_ADD = (
    "<SEG_A> <ADD><DX><OUT><ALPHA> Additionner, c'est accueillir </SEG_A> "
    "<SEG_B> <ADD><DX><OUT><OMEGA> ce qui vient s'adjoindre, </SEG_B> "
    "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> sans dissoudre ce qui demeure.<EOL> "
    "</SEG_A_PRIME> <EOL>"
)
LINE_MUL = (
    "<SEG_A> <MUL><DX><OUT><ALPHA> Xⁿ est l'itération de </SEG_A> "
    "<SEG_B> <MUL><DX><OUT><OMEGA> X, chaque n altérant </SEG_B> "
    "<SEG_A_PRIME> <MUL><LV><IN><A_PRIME> son spectre et sa résonance.<EOL> "
    "</SEG_A_PRIME> <EOL>"
)
TERMINATOR_STRUCT = (
    "<SEG_A> <EOS> </SEG_A> <SEG_B> </SEG_B> <SEG_A_PRIME> </SEG_A_PRIME> <EOL>"
)


def test_parse_basic_cycle() -> None:
    cyc = parse_aba_line(LINE_ADD)
    assert cyc.op == "ADD"
    assert cyc.seg_a.text == "Additionner, c'est accueillir"
    assert cyc.seg_b.text == "ce qui vient s'adjoindre,"
    assert cyc.seg_a_prime.text == "sans dissoudre ce qui demeure."
    # En-têtes de tags.
    assert (cyc.seg_a.chirality, cyc.seg_a.direction, cyc.seg_a.position) == ("DX", "OUT", "ALPHA")
    assert (cyc.seg_b.position, cyc.seg_b.direction) == ("OMEGA", "OUT")
    assert (cyc.seg_a_prime.chirality, cyc.seg_a_prime.direction, cyc.seg_a_prime.position) == ("LV", "IN", "A_PRIME")


def test_closure_invariant() -> None:
    cyc = parse_aba_line(LINE_MUL)
    assert cyc.is_closure  # DX/OUT, DX/OUT, LV/IN
    assert validate_cycle(cyc) == []  # conforme au canon


def test_triplet_shape() -> None:
    t = parse_aba_line(LINE_ADD).triplet
    assert t["op"] == "ADD"
    assert set(t["segments"].keys()) == {"SEG_A", "SEG_B", "SEG_A_PRIME"}
    assert t["chirality"]["SEG_A_PRIME"] == "LV"
    assert t["direction"]["SEG_A"] == "OUT"


def test_terminator_detection() -> None:
    assert is_terminator_line("<EOS>")
    assert is_terminator_line(TERMINATOR_STRUCT)
    assert try_parse_aba_line("<EOS>") is None
    assert try_parse_aba_line(TERMINATOR_STRUCT) is None
    assert try_parse_aba_line("   ") is None


def test_fixed_point_detection() -> None:
    """Cycle répété à l'identique = point fixe (l'écho du vide)."""
    same = "L'écho du vide propage l'information."
    line = (
        f"<SEG_A> <MUL><DX><OUT><ALPHA> {same}<EOL> </SEG_A> "
        f"<SEG_B> <MUL><DX><OUT><OMEGA> {same}<EOL> </SEG_B> "
        f"<SEG_A_PRIME> <MUL><LV><IN><A_PRIME> {same}<EOL> </SEG_A_PRIME> <EOL>"
    )
    cyc = parse_aba_line(line)
    assert cyc.is_fixed_point


def test_inconsistent_operator_flagged() -> None:
    """Opérateur non constant sur le cycle => écart signalé (mais parsé)."""
    line = (
        "<SEG_A> <ADD><DX><OUT><ALPHA> a </SEG_A> "
        "<SEG_B> <SUB><DX><OUT><OMEGA> b </SEG_B> "
        "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> c </SEG_A_PRIME> <EOL>"
    )
    cyc = parse_aba_line(line)
    issues = validate_cycle(cyc)
    assert any("opérateur non constant" in s for s in issues)


def test_valid_cycle_with_literal_eos_in_text_not_dropped() -> None:
    """Régression : un cycle valide dont le texte contient « <EOS> » n'est pas
    confondu avec un terminateur (il a, lui, des en-têtes d'opérateur valides)."""
    line = (
        "<SEG_A> <ADD><DX><OUT><ALPHA> le token <EOS> marque la fin </SEG_A> "
        "<SEG_B> <ADD><DX><OUT><OMEGA> b </SEG_B> "
        "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> c </SEG_A_PRIME> <EOL>"
    )
    assert not is_terminator_line(line)
    cyc = try_parse_aba_line(line)
    assert cyc is not None
    assert cyc.op == "ADD"
    # Le tag <EOS> est nettoyé du texte, mais le cycle n'est PAS jeté.
    assert "fin" in cyc.seg_a.text


def test_malformed_raises() -> None:
    with pytest.raises(AbaParseError):
        parse_aba_line("<SEG_A> pas d'en-tête </SEG_A> <SEG_B> </SEG_B>")
    with pytest.raises(AbaParseError):
        parse_aba_line("ceci n'est pas une ligne ABA")
