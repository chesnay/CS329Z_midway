"""Put flattened Codeforces math back into a form that keeps the formula.

open-r1 stored statement text already lost the LaTeX. A power such as
``10^{18}`` was saved as ``1018``, and a subscript such as ``x_{n-1}`` was
saved as ``xn - 1``. Simple indexes such as ``a1`` for ``a_1`` still read as
indexes, so they are left alone. A literal integer, a binary code, and an
explicit ``$$...$$`` formula are left alone too.
"""

import re

import pyarrow as pa

# Statement prose. Sample examples, official tests, and the editorial stay as stored.
STATEMENT_FIELDS = (
    "description",
    "input_format",
    "output_format",
    "note",
    "interaction_format",
)

# Longer exponents first so 1018 is 10^18, not 10^1 followed by 018.
_EXP = r"18|15|12|10|9|8|7|6|5|4|3|2"
_POW = rf"(?<![\d.])10({_EXP})(?!\d)"
_CMP = r">=|<=|≥|≤|>|<"
_MINUS = r"[-–—−]"
_PLUS = r"[+]"
_SIGN = rf"(?:{_PLUS}|{_MINUS})"

_DOLLAR = re.compile(r"\$\$.*?\$\$", re.DOTALL)


def restore_statement_math(text):
    """Return statement text with the flattened powers and subscripts repaired.

    The same string is returned when it is applied again. Sample inputs and
    sample outputs are not statement text; callers should not pass them here.
    """
    if not text:
        return text or ""
    masked, chunks = _mask_dollars(text)
    masked = _restore_subscripts(masked)
    masked = _restore_powers(masked)
    return _unmask_dollars(masked, chunks)


def restore_statement_table(table):
    """Return the problems table with statement columns repaired.

    The other columns are the same arrays. The two counts are repaired
    fields and repaired problems. Applying this again changes nothing.
    """
    ids = table.column("id").to_pylist() if "id" in table.column_names else [None] * table.num_rows
    arrays = []
    fields_changed = 0
    problems_changed = set()
    for name in table.column_names:
        column = table.column(name)
        if name not in STATEMENT_FIELDS:
            arrays.append(column)
            continue
        repaired = []
        for problem_id, value in zip(ids, column.to_pylist()):
            if not value:
                repaired.append(value)
                continue
            new = restore_statement_math(value)
            repaired.append(new)
            if new != value:
                fields_changed += 1
                problems_changed.add(problem_id)
        arrays.append(pa.array(repaired, type=column.type))
    return pa.Table.from_arrays(arrays, schema=table.schema), fields_changed, len(problems_changed)


def _mask_dollars(text):
    chunks = []

    def hold(match):
        chunks.append(match.group(0))
        return f"\0MATH{len(chunks) - 1}\0"

    return _DOLLAR.sub(hold, text), chunks


def _unmask_dollars(text, chunks):
    for index, chunk in enumerate(chunks):
        text = text.replace(f"\0MATH{index}\0", chunk)
    return text


def _sub(base, index):
    return f"{base}_{{{index}}}"


def _sign(token):
    return "+" if token == "+" else "-"


def _restore_subscripts(text):
    text = _TEL.sub(_tel, text)
    text = _LEX.sub(_lex, text)
    text = _PEAK_UP.sub(_peak, text)
    text = _PEAK_DOWN.sub(_peak, text)
    text = _RECUR.sub(_recur, text)
    text = _TRIPLE.sub(_triple, text)
    text = _PAGE.sub(_page, text)
    text = _CALL.sub(_call, text)
    text = _SWAP.sub(_swap, text)
    text = _LIST.sub(_list_term, text)
    text = _CHAIN.sub(_chain_term, text)
    text = _GEOM.sub(_geom, text)
    text = _ABS.sub(_abs_pair, text)
    text = _QUOTE.sub(_quote, text)
    text = _NEXT_IS.sub(_next_is, text)
    text = _POW_TAIL.sub(_pow_tail, text)
    text = _DIGIT_POW.sub(_digit_pow, text)
    pairs = {(m.group("b"), m.group("idx"), _sign(m.group("op")), m.group("k")) for m in _SAME.finditer(text)}
    text = _SAME.sub(_same, text)
    return _companions(text, pairs)


def _restore_powers(text):
    text = _INEQ.sub(_keep_pow, text)
    text = _NEG_BOUND.sub(_keep_pow, text)
    text = _CDOT.sub(_keep_pow, text)
    text = _MODULO.sub(_keep_pow, text)
    text = _PAREN_POW.sub(_paren_pow, text)
    text = _MAGNITUDE.sub(_keep_pow, text)
    text = _MIN_POW.sub(_keep_pow, text)
    text = _SUM_POW.sub(_keep_pow, text)
    text = _SQUARE_EQ.sub(_square_eq, text)
    text = _ERROR_POW.sub(_error_pow, text)
    return text


def _keep_pow(match):
    """Replace the 10-and-exponent number and keep the surrounding characters."""
    number = match.group("num")
    start = match.start("num") - match.start()
    end = match.end("num") - match.start()
    body = match.group(0)
    return body[:start] + f"10^{{{number[2:]}}}" + body[end:]


_TEL = re.compile(
    rf"\((?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)\s*{_MINUS}\s*(?P=b)(?P=idx)\)"
)


def _tel(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"({_sub(base, index + '-' + k)} - {_sub(base, index)})"


_LEX = re.compile(
    rf"(?P<eq>(?P<b1>[A-Za-z])(?P<idx>[A-Za-z]+)\s*=\s*(?P<b2>[A-Za-z])(?P=idx)\s+and\s+)"
    rf"(?P=b1)(?P=idx)\s*{_PLUS}\s*(?P<k>\d+)\s*(?P<cmp>{_CMP})\s*"
    rf"(?P=b2)(?P=idx)\s*{_PLUS}\s*(?P=k)(?!\d)"
)


def _lex(match):
    base1, base2 = match.group("b1"), match.group("b2")
    index, k = match.group("idx"), match.group("k")
    return (
        f"{match.group('eq')}{_sub(base1, index + '+' + k)} "
        f"{match.group('cmp')} {_sub(base2, index + '+' + k)}"
    )


def _peak_pattern(left, right):
    return re.compile(
        rf"(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)\s*"
        rf"(?P<left>{left})\s*(?P=b)(?P=idx)\s*(?P<right>{right})\s*(?P=b)(?P=idx)\s*"
        rf"{_PLUS}\s*(?P=k)(?!\d)"
    )


_PEAK_UP = _peak_pattern("<", ">")
_PEAK_DOWN = _peak_pattern(">", "<")


def _peak(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    middle = f"{base}{index}"
    return f"{_sub(base, index + '-' + k)} {match.group('left')} {middle} {match.group('right')} {_sub(base, index + '+' + k)}"


_RECUR = re.compile(
    rf"(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*=\s*(?P=b)(?P=idx)\s*{_MINUS}\s*(?P<k1>\d+)\s*"
    rf"{_PLUS}\s*(?P=b)(?P=idx)\s*{_MINUS}\s*(?P<k2>\d+)(?!\d)"
)


def _recur(match):
    base, index = match.group("b"), match.group("idx")
    return (
        f"{_sub(base, index)} = {_sub(base, index + '-' + match.group('k1'))} "
        f"+ {_sub(base, index + '-' + match.group('k2'))}"
    )


_TRIPLE = re.compile(
    rf"(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_PLUS}\s*(?P=b)(?P=idx)\s*{_PLUS}\s*(?P<step>[A-Za-z]+)\s*"
    rf"{_PLUS}\s*(?P=b)(?P=idx)\s*{_PLUS}\s*(?P<coef>\d+)(?P=step)"
)


def _triple(match):
    base, index, step, coef = match.group("b"), match.group("idx"), match.group("step"), match.group("coef")
    return f"{_sub(base, index)} + {_sub(base, index + '+' + step)} + {_sub(base, index + '+' + coef + step)}"


# r_i + 1 < l_{i+1}. The left plus is real addition; only the right index contains the plus.
_PAGE = re.compile(rf"r(?P<idx>[A-Za-z]+)\s*{_PLUS}\s*(?P<k>\d+)\s*<\s*l(?P=idx)\s*{_PLUS}\s*(?P=k)(?!\d)")


def _page(match):
    index, k = match.group("idx"), match.group("k")
    return f"r{index} + {k} < {_sub('l', index + '+' + k)}"


_CALL = re.compile(
    rf"(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*=\s*f\(\s*(?P=b)(?P=idx)\s*{_MINUS}\s*(?P<k>\d+)\s*\)"
)


def _call(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"{_sub(base, index)} = f({_sub(base, index + '-' + k)})"


_SWAP = re.compile(
    rf"swap\(\s*(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)\s*,\s*(?P=b)(?P=idx)\s*\)"
)


def _swap(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"swap({_sub(base, index + '-' + k)}, {_sub(base, index)})"


_LIST = re.compile(
    rf"(?P<b>[A-Za-z])1(?P<gap1>\s*,\s*)(?P=b)2(?P<gap2>\s*,\s*(?:\.\.\.|…)\s*,?\s*)"
    rf"(?P=b)(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)(?!\d)"
)


def _list_term(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"{base}1{match.group('gap1')}{base}2{match.group('gap2')}{_sub(base, index + '-' + k)}"


_CHAIN = re.compile(
    rf"(?P<b>[A-Za-z])1(?P<gap1>\s*<\s*)(?P=b)2(?P<gap2>\s*<\s*(?:\.\.\.|…)\s*<\s*)"
    rf"(?P=b)(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)(?!\d)"
)


def _chain_term(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"{base}1{match.group('gap1')}{base}2{match.group('gap2')}{_sub(base, index + '-' + k)}"


_GEOM = re.compile(
    rf"(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*=\s*(?P=b)(?P=idx)\s*{_MINUS}\s*(?P<k>\d+)"
    rf"(?P<factor>[A-Za-z])(?![A-Za-z])"
)


def _geom(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"{_sub(base, index)} = {_sub(base, index + '-' + k)} {match.group('factor')}"


_ABS = re.compile(
    rf"\|(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P=b)(?P=idx)\s*{_PLUS}\s*(?P<k>\d+)\|"
)


def _abs_pair(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    return f"|{_sub(base, index)} - {_sub(base, index + '+' + k)}|"


_QUOTE = re.compile(rf"[\"“](?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)[\"”]")


def _quote(match):
    return f'"{_sub(match.group("b"), match.group("idx") + "-" + match.group("k"))}"'


# The index is one or two letters. A longer word, such as "position + 3 is", is English.
_NEXT_IS = re.compile(
    rf"(?<![A-Za-z])(?P<b>[A-Za-z])(?P<idx>[A-Za-z]{{1,2}})\s*{_PLUS}\s*(?P<k>\d+)\s+is\b"
)


def _next_is(match):
    return f"{_sub(match.group('b'), match.group('idx') + '+' + match.group('k'))} is"


# x^{p-2} - 1, where the flattened superscript was "xp - 2 - 1".
_POW_TAIL = re.compile(
    rf"(?<![A-Za-z0-9])(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*{_MINUS}\s*(?P<k>\d+)\s*{_MINUS}\s*1(?!\d)"
)


def _pow_tail(match):
    return f"{match.group('b')}^{{{match.group('idx')}-{match.group('k')}}} - 1"


# x^2 - 1 written "x2 - 1" immediately before an ellipsis of the same list.
_DIGIT_POW = re.compile(rf"(?<![A-Za-z0-9])(?P<b>[A-Za-z])(?P<k>\d+)\s*{_MINUS}\s*1(?=\s*,\s*(?:\.\.\.|…))")


def _digit_pow(match):
    return f"{match.group('b')}^{{{match.group('k')}}} - 1"


_SAME = re.compile(
    rf"(?<![A-Za-z])(?P<b>[A-Za-z])(?P<idx>[A-Za-z]+)\s*(?P<cmp>{_CMP})\s*"
    rf"(?P=b)(?P=idx)\s*(?P<op>{_SIGN})\s*(?P<k>\d+)(?!\d)"
)


def _same(match):
    base, index, k = match.group("b"), match.group("idx"), match.group("k")
    op = _sign(match.group("op"))
    return f"{_sub(base, index)} {match.group('cmp')} {_sub(base, index + op + k)}"


def _companions(text, pairs):
    """A comparison already showed that ``Xi - 1`` means ``X_{i-1}``.

    Other copies of that same token in the field are the same subscript.
    ``X_i - 1`` as real subtraction does not occur beside ``X_i ≥ X_i - 1``,
    because that comparison would be true for every number.
    """
    for base, index, op, k in pairs:
        sign = _PLUS if op == "+" else _MINUS
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(base + index)}\s*{sign}\s*{k}(?!\d)"
        )
        text = pattern.sub(_sub(base, index + op + k), text)
    return text


_INEQ = re.compile(rf"(?:<=|>=|≤|≥|<|>)\s*(?:-\s*)?{_POW}")
_NEG_BOUND = re.compile(rf"(?<![\w.])-\s*{_POW}(?=\s*(?:<=|>=|≤|≥|<))")
_CDOT = re.compile(rf"[·⋅×*]\s*{_POW}")
_MODULO = re.compile(rf"(?i)\b(?:modulo|modulus)\s+{_POW}\s*\+\s*\d+")
_MAGNITUDE = re.compile(
    rf"(?i)\b(?:exceed(?:s|ed|ing)?|at most|at least|(?:no\s+)?greater than|(?:no\s+)?less than)\s+{_POW}"
)
_MIN_POW = re.compile(rf"(?i)\bmin\s*[{{\(]\s*{_POW}")
_SUM_POW = re.compile(rf"(?i)\bsum\s+{_POW}\s+numbers\b")
_PAREN_POW = re.compile(rf"(\d{{5,}})\s*\(\s*{_POW}(\s*\+\s*(\d+))?\s*\)")
_SQUARE_EQ = re.compile(rf"{_POW}(?=\s*\+\s*\d+\s*[·⋅×*]\s*10\b)")
_ERROR_POW = re.compile(
    rf"(?i)\b(?:error|precision)\b[^.\n]{{0,80}}?10\s*{_MINUS}\s*(\d{{1,2}})(?!\d)"
)


def _paren_pow(match):
    number = match.group("num")
    exp = int(number[2:])
    added = int(match.group(5) or 0)
    if int(match.group(1)) != 10**exp + added:
        return match.group(0)
    start = match.start("num") - match.start()
    end = match.end("num") - match.start()
    body = match.group(0)
    return body[:start] + f"10^{{{number[2:]}}}" + body[end:]


def _square_eq(match):
    return f"10^{{{match.group('num')[2:]}}}"


def _error_pow(match):
    exponent = match.group(1)
    body = match.group(0)
    return re.sub(rf"10\s*{_MINUS}\s*{exponent}$", f"10^{{-{exponent}}}", body)


# The named group is attached to the shared power expression.
def _with_num(pattern):
    return pattern.pattern.replace(_POW, f"(?P<num>{_POW})", 1)


_INEQ = re.compile(_with_num(_INEQ))
_NEG_BOUND = re.compile(_with_num(_NEG_BOUND))
_CDOT = re.compile(_with_num(_CDOT))
_MODULO = re.compile(_with_num(_MODULO))
_MAGNITUDE = re.compile(_with_num(_MAGNITUDE))
_MIN_POW = re.compile(_with_num(_MIN_POW))
_SUM_POW = re.compile(_with_num(_SUM_POW))
_PAREN_POW = re.compile(_with_num(_PAREN_POW))
_SQUARE_EQ = re.compile(_with_num(_SQUARE_EQ))
