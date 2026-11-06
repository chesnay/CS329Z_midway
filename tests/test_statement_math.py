"""Flattened Codeforces math is restored only where the formula changed."""

import pyarrow as pa

from tagent.data.statement_math import restore_statement_math, restore_statement_table


def test_power_of_ten_in_a_constraint():
    text = "The only line contains one integer n (1 ≤ n ≤ 1018)."
    assert restore_statement_math(text) == "The only line contains one integer n (1 ≤ n ≤ 10^{18})."


def test_negative_bound_and_cdot_and_modulo():
    text = "x ( - 109 ≤ x ≤ 109), n ≤ 2·1018, print it modulo 109 + 7."
    restored = restore_statement_math(text)
    assert restored == "x ( - 10^{9} ≤ x ≤ 10^{9}), n ≤ 2·10^{18}, print it modulo 10^{9} + 7."


def test_verified_parenthetical_modulus_and_error_tolerance():
    text = "modulo 1000000007 (109 + 7). The absolute error must not exceed 10 - 6."
    restored = restore_statement_math(text)
    assert "1000000007 (10^{9} + 7)" in restored
    assert "10^{-6}" in restored


def test_last_term_of_a_difference_row_and_the_next_index():
    text = "(x1 - x2) + (x2 - x3) + ... + (xn - 1 - xn). xr = yr and xr + 1 < yr + 1."
    restored = restore_statement_math(text)
    assert "(x_{n-1} - x_{n})" in restored
    assert "(x1 - x2)" in restored
    assert "x_{r+1} < y_{r+1}" in restored


def test_adjacent_index_and_recurrence():
    assert "a_{i} ≥ a_{i+1}" in restore_statement_math("ai ≥ ai + 1")
    assert "F_{i} = F_{i-2} + F_{i-1}" in restore_statement_math("Fi = Fi - 2 + Fi - 1")
    assert "|p_{i} - p_{i+1}|" in restore_statement_math("|pi–pi + 1|")
    assert "y_{j-1} < yj > y_{j+1}" in restore_statement_math("yj - 1 < yj > yj + 1")


def test_page_number_keeps_the_real_plus_on_the_left():
    text = "where ri + 1 < li + 1 for all i"
    assert restore_statement_math(text) == "where ri + 1 < l_{i+1} for all i"


def test_literal_numbers_and_binary_codes_stay():
    samples = [
        "integers 100, 99 and 102",
        "numbers 102, 103, 104, 105, 106, 107, 108, 109",
        '">" → 1010, "-" → 1011',
        "510 = 1012 and 1010 = 10102",
        "lowbit(100102) = 102",
        "a year that consists of 108 days",
        "print -1",
        "Codecraft-17",
        "nm + 1 lines",
        "the temperature is  - 10 - 5 =  - 15",
        "(10 - 1 ≥ 9)",
        "x1 - x2",
        "the correct position  + 3 is 0",
    ]
    for sample in samples:
        assert restore_statement_math(sample) == sample


def test_explicit_latex_is_left_alone():
    text = "n ($$$1 \\le n \\le 10^9$$$) and also n ≤ 1018"
    restored = restore_statement_math(text)
    assert "$$$1 \\le n \\le 10^9$$$" in restored
    assert "10^{18}" in restored


def test_restoring_twice_changes_nothing():
    text = "(xn - 1 - xn) and n ≤ 1018 and modulo 109 + 7 and error 10 - 9"
    once = restore_statement_math(text)
    assert restore_statement_math(once) == once


def test_table_repairs_statement_fields_and_keeps_samples():
    table = pa.table({
        "id": ["630/K"],
        "description": ["n ≤ 1018"],
        "input_format": ["1 ≤ n ≤ 1018"],
        "output_format": [None],
        "note": ["ai ≥ ai + 1"],
        "interaction_format": [None],
        "examples": [[{"input": "1018", "output": "2"}]],
        "editorial": ["still 1018"],
    })
    restored, fields_changed, problems_changed = restore_statement_table(table)
    assert restored.column("input_format")[0].as_py() == "1 ≤ n ≤ 10^{18}"
    assert restored.column("note")[0].as_py() == "a_{i} ≥ a_{i+1}"
    assert restored.column("examples")[0].as_py() == [{"input": "1018", "output": "2"}]
    assert restored.column("editorial")[0].as_py() == "still 1018"
    assert fields_changed == 3
    assert problems_changed == 1
    again, fields_again, problems_again = restore_statement_table(restored)
    assert fields_again == 0
    assert problems_again == 0
    assert again.column("input_format")[0].as_py() == "1 ≤ n ≤ 10^{18}"
