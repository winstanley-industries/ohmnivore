"""Bounded, local-only expansion of the exact EMI-01 SiC model closure.

No vendor equations are distributed by this module. Input equations are verified
by the original archive adapter and flattened only in caller-owned scratch space.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import tempfile

from reference.emi01 import adapter, circuits

VERSION = "emi02-microchip-import-v1"
# ngspice46 ps reserves vt even when a local parameter has the same name.
# This freezes its actual ambient-temperature frontend binding (ADR-006).
REFERENCE_AMBIENT_C = 27.0
REFERENCE_THERMAL_VOLTAGE = (REFERENCE_AMBIENT_C + 273.15) * 8.6173303e-5
PACKAGES = {"msc040sma120b", "mscsicfet1200"}
MAX_PACKAGES = 4
MAX_INSTANCES = 16
MAX_DEPTH = 3
MAX_UNKNOWNS = 512
MAX_EXPRESSION_NODES = 512
MAX_TOTAL_NODES = 16384
MAX_EXPRESSION_DEPTH = 64
IDENT = r"[a-zA-Z_][a-zA-Z_0-9]*"
NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:meg|[tgkmunpf])?"
TOKEN = re.compile(rf"\s*({NUMBER}|{IDENT}|\*\*|[+*/(){{}}-])", re.I)
SCALE = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "": 1.0,
}


class ImportError(ValueError):
    """A typed import failure; contains no proprietary equation text."""

    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status


def fail(status, message):
    raise ImportError(status, message)


def number(token):
    match = re.fullmatch(
        r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(meg|[tgkmunpf])?",
        token,
        re.I,
    )
    if not match:
        fail("unsupported_input", "unsupported numeric literal")
    value = float(match[1]) * SCALE[(match[2] or "").lower()]
    if not math.isfinite(value):
        fail("non_finite", "non-finite parameter")
    return value


class ParameterExpression:
    """Strict selected numparam subset, separately evaluated from runtime laws."""

    def __init__(self, expression, lookup):
        self.tokens = []
        position = 0
        expression = expression.strip()
        while position < len(expression):
            match = TOKEN.match(expression, position)
            if not match:
                fail("unsupported_input", "unsupported parameter expression token")
            self.tokens.append(match[1])
            position = match.end()
        if len(self.tokens) > MAX_EXPRESSION_NODES:
            fail("resource_limit", "parameter expression node budget")
        self.position = 0
        self.lookup = lookup
        self.depth = 0

    def pop(self):
        if self.position == len(self.tokens):
            fail("parse_failure", "incomplete parameter expression")
        result = self.tokens[self.position]
        self.position += 1
        return result

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else ""

    def atom(self):
        self.depth += 1
        if self.depth > MAX_EXPRESSION_DEPTH:
            fail("resource_limit", "parameter expression depth budget")
        token = self.pop()
        if token in ("+", "-"):
            result = self.atom() * (-1 if token == "-" else 1)
        elif token in ("(", "{"):
            result = self.expression()
            if self.pop() != (")" if token == "(" else "}"):
                fail("parse_failure", "unbalanced parameter expression")
        elif token.lower() == "exp" and self.peek() == "(":
            self.pop()
            result = math.exp(self.expression())
            if self.pop() != ")":
                fail("parse_failure", "unbalanced exponential")
        elif re.fullmatch(IDENT, token):
            result = self.lookup(token.lower())
        else:
            result = number(token)
        self.depth -= 1
        return result

    def product(self):
        result = self.atom()
        while self.peek() in ("*", "/"):
            operator, rhs = self.pop(), self.atom()
            result = result * rhs if operator == "*" else result / rhs
        return result

    def expression(self):
        result = self.product()
        while self.peek() in ("+", "-"):
            operator, rhs = self.pop(), self.product()
            result = result + rhs if operator == "+" else result - rhs
        return result

    def evaluate(self):
        try:
            value = self.expression()
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            if isinstance(exc, ImportError):
                raise
            fail("non_finite", "invalid parameter arithmetic")
        if self.position != len(self.tokens):
            fail("unsupported_input", "unsupported parameter operation")
        if not math.isfinite(value):
            fail("non_finite", "non-finite parameter evaluation")
        return value


def assignments(text):
    """Split balanced assignments without evaluating or ignoring any bytes."""
    result = {}
    position = 0
    while position < len(text):
        match = re.match(rf"\s*({IDENT})\s*=\s*", text[position:])
        if not match:
            if not text[position:].strip():
                break
            fail("parse_failure", "invalid parameter assignment")
        name = match[1].lower()
        position += match.end()
        start, depth = position, 0
        while position < len(text):
            char = text[position]
            if char in "({":
                depth += 1
            elif char in ")}":
                depth -= 1
                if depth < 0:
                    fail("parse_failure", "unbalanced parameter assignment")
            if depth == 0 and char.isspace():
                rest = text[position:]
                if re.match(rf"\s*{IDENT}\s*=", rest):
                    break
            position += 1
        value = text[start:position].strip()
        if depth or not value or name in result:
            fail("parse_failure", "duplicate or malformed parameter assignment")
        result[name] = value
    return result


def resolve_parameters(definitions, caller=None, overrides=None):
    definitions = {key.lower(): value for key, value in definitions.items()}
    caller = {} if caller is None else {k.lower(): v for k, v in caller.items()}
    overrides = {} if overrides is None else overrides
    resolved, active = {}, set()

    def lookup_caller(name):
        if name not in caller:
            fail("unsupported_input", "unknown caller parameter")
        return caller[name]

    for name, expression in overrides.items():
        name = name.lower()
        if name not in definitions:
            fail("unsupported_input", "unknown parameter override")
        resolved[name] = ParameterExpression(expression, lookup_caller).evaluate()

    def lookup(name):
        if name in resolved:
            return resolved[name]
        if name in active:
            fail("unsupported_input", "cyclic parameter dependency")
        if name not in definitions:
            fail("unsupported_input", "unknown local parameter")
        active.add(name)
        resolved[name] = ParameterExpression(definitions[name], lookup).evaluate()
        active.remove(name)
        return resolved[name]

    for name in definitions:
        lookup(name)
    return resolved


def logical_lines(text):
    lines = []
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            if not lines:
                fail("parse_failure", "orphan model continuation")
            lines[-1] += " " + line[1:].strip()
        else:
            lines.append(line)
    return lines


@dataclass
class Definition:
    terminals: list[str]
    defaults: dict[str, str]
    parameters: dict[str, str]
    cards: list[str]


def definitions(text):
    result, current = {}, None
    for line in logical_lines(text):
        fields = line.split()
        if fields[0].lower() == ".subckt":
            if current is not None:
                fail("unsupported_input", "nested model declaration")
            name = fields[1].lower()
            current = name
            if name in PACKAGES:
                if name in result:
                    fail("unsupported_input", "duplicate selected model")
                match = re.fullmatch(
                    r"\.subckt\s+\S+\s+(.*?)\s+params:\s*(.*)", line, re.I
                )
                if not match:
                    fail("unsupported_input", "selected model header changed")
                result[name] = Definition(
                    match[1].lower().split(), assignments(match[2]), {}, []
                )
        elif fields[0].lower() == ".ends":
            current = None
        elif current in PACKAGES:
            d = result[current]
            if fields[0].lower() == ".param":
                addition = assignments(line[len(fields[0]) :])
                if set(addition) & (set(d.defaults) | set(d.parameters)):
                    fail("unsupported_input", "duplicate selected parameter")
                d.parameters.update(addition)
            else:
                d.cards.append(line)
    if set(result) != PACKAGES or current is not None:
        fail("unsupported_input", "selected model closure incomplete")
    return result


def component_name(scope, original):
    original = original.lower()
    return original[0] + "_" + "_".join(scope) + "__" + original[1:]


def bind_expression(expression, nodes, scope, parameters):
    """Alpha-rename only nodes, sensed source names and bound parameters."""

    def voltage(match):
        values = [value.strip().lower() for value in match[1].split(",")]
        if len(values) not in (1, 2) or any(
            not re.fullmatch(r"[a-z0-9_]+", v) for v in values
        ):
            fail("unsupported_input", "invalid voltage control")
        return "v(" + ",".join(nodes(v) for v in values) + ")"

    def current(match):
        value = match[1].strip().lower()
        if not re.fullmatch(r"v[a-z0-9_]+", value):
            fail("unsupported_input", "invalid sensing-source control")
        return "i(" + component_name(scope, value) + ")"

    # Placeholders protect bound node and source identifiers from parameter replacement.
    bound = []

    def protect(match):
        value = voltage(match) if match[0][0].lower() == "v" else current(match)
        bound.append(value)
        return f"@{len(bound) - 1}@"

    expression = re.sub(r"[vi]\s*\(([^()]*)\)", protect, expression, flags=re.I)

    def parameter(match):
        token = match[0]
        key = token.lower()
        if key in parameters:
            return "(" + format(parameters[key], ".17g") + ")"
        if key in {"exp", "if"}:
            return key
        # Exponents and engineering literals are consumed together by this regex.
        if re.fullmatch(NUMBER, token, re.I):
            return token
        fail("unsupported_input", "unbound expression identifier")

    expression = re.sub(rf"{NUMBER}|{IDENT}", parameter, expression, flags=re.I)
    expression = re.sub(r"@([0-9]+)@", lambda m: bound[int(m[1])], expression)
    return expression


def flatten(deck, adapted):
    """Expand exact adapted bytes; public caller must additionally verify archive."""
    if hashlib.sha256(adapted).hexdigest() != adapter.ADAPTED_SHA256:
        fail("provenance_mismatch", "adapted model hash mismatch")
    if len(deck.encode("utf-8")) > 1024 * 1024:
        fail("resource_limit", "reference deck byte budget")
    model = definitions(adapted.decode("utf-8"))
    lines = logical_lines(deck)
    if not lines:
        fail("parse_failure", "empty reference deck")
    output = ["* " + lines[0]]
    counters = {"packages": 0, "instances": 0, "expression_tokens": 0}
    native_names, node_names, branch_names = set(), set(), set()

    def append(card):
        fields = card.split()
        identity = fields[0].lower()
        if identity in native_names:
            fail("compile_failure", "duplicate expanded component")
        native_names.add(identity)
        if identity[0] != "k":
            node_names.update(
                n.lower() for n in fields[1:3] if n.lower() not in {"0", "gnd"}
            )
        if identity[0] in {"v", "l", "e"}:
            branch_names.add(identity)
        if len(node_names) + len(branch_names) > MAX_UNKNOWNS:
            fail("resource_limit", "expanded MNA unknown budget")
        output.append(card)

    def expand(card, parent_nodes, parent_scope, caller, depth):
        fields = card.split()
        if depth > MAX_DEPTH:
            fail("resource_limit", "subcircuit depth budget")
        counters["instances"] += 1
        if counters["instances"] > MAX_INSTANCES:
            fail("resource_limit", "subcircuit instance budget")
        if len(fields) < 5:
            fail("parse_failure", "short package instance")
        name = fields[4].lower()
        if name not in PACKAGES or (depth == 1 and name != "msc040sma120b"):
            fail("unsupported_input", "unsupported package")
        if depth > 1 and name != "mscsicfet1200":
            fail("unsupported_input", "unsupported model recursion")
        if depth == 1:
            counters["packages"] += 1
            if counters["packages"] > MAX_PACKAGES:
                fail("resource_limit", "package count budget")
        d = model[name]
        overrides = {}
        if len(fields) > 5:
            match = re.fullmatch(
                r"\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+params:\s*(.*)", card, re.I
            )
            if not match:
                fail("unsupported_input", "unknown instance suffix")
            overrides = assignments(match[1])
        if depth == 1 and set(overrides) - {"tj_c"}:
            fail(
                "unsupported_input", "only fixed junction temperature may be overridden"
            )
        parameter_graph = {**d.defaults, **d.parameters}
        if "vt" in parameter_graph:
            parameter_graph["vt"] = format(REFERENCE_THERMAL_VOLTAGE, ".17g")
        parameters = resolve_parameters(parameter_graph, caller, overrides)
        if parameters["tj_c"] not in {27.0, 125.0}:
            fail("unsupported_input", "unsupported fixed junction temperature")
        scope = parent_scope + [fields[0].lower()]
        if not all(re.fullmatch(IDENT, s) for s in scope):
            fail("unsupported_input", "unsupported instance identifier")
        terminal_map = dict(
            zip(
                d.terminals, [parent_nodes(n.lower()) for n in fields[1:4]], strict=True
            )
        )

        def nodes(node):
            node = node.lower()
            if node in {"0", "gnd"}:
                return "0"
            return terminal_map.get(node, "emi02_" + "_".join(scope) + "__" + node)

        for nested in d.cards:
            parts = nested.split(maxsplit=3)
            kind = parts[0][0].lower()
            if kind == "x":
                expand(nested, nodes, scope, parameters, depth + 1)
                continue
            if len(parts) != 4 or kind not in "rlcvegb":
                fail("unsupported_input", "unsupported selected model card")
            prefix = " ".join(
                [component_name(scope, parts[0]), nodes(parts[1]), nodes(parts[2])]
            )
            if kind in "rlcv":
                value = ParameterExpression(
                    parts[3],
                    lambda n: (
                        parameters[n]
                        if n in parameters
                        else fail("unsupported_input", "unknown passive parameter")
                    ),
                ).evaluate()
                append(prefix + " " + format(value, ".17g"))
            else:
                match = re.fullmatch(r"(?:value|i)\s*=\s*(\{.*\})", parts[3], re.I)
                if not match or (
                    kind == "b" and not parts[3].lower().lstrip().startswith("i")
                ):
                    fail("unsupported_input", "unsupported behavioral card")
                expression = bind_expression(match[1], nodes, scope, parameters)
                count = len(
                    re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|[0-9.]+|\S", expression)
                )
                if count > MAX_EXPRESSION_NODES:
                    fail("resource_limit", "behavioral expression node budget")
                counters["expression_tokens"] += count
                if counters["expression_tokens"] > MAX_TOTAL_NODES:
                    fail("resource_limit", "total expression budget")
                append(prefix + (" I=" if kind == "b" else " VALUE=") + expression)

    seen = set()
    for line in lines[1:]:
        if ".end" in seen:
            fail("parse_failure", "cards after circuit end")
        fields = line.split()
        key = fields[0].lower()
        if (
            not key.startswith(".")
            and key[0] != "k"
            and any(n.lower().startswith("emi02_") for n in fields[1:4])
        ):
            fail("unsupported_input", "reserved internal node namespace")
        if key.startswith("."):
            if key in seen:
                fail("parse_failure", "duplicate reference directive")
            seen.add(key)
            if key == ".include" and fields == [".include", "model.lib"]:
                continue
            if key == ".options" and line == circuits.OPTIONS:
                continue
            if key == ".tran" and len(fields) == 5:
                start, stop, begin, maximum = map(number, fields[1:])
                if start <= 0 or maximum <= 0 or stop <= 0 or begin != 0:
                    fail("unsupported_input", "invalid reference transient bounds")
                output.append(f".tran {maximum:.17g} {stop:.17g} 0")
                continue
            if key in {".save", ".end", ".op"}:
                output.append(line)
                continue
            fail("unsupported_input", "unsupported reference directive")
        if key[0] == "x":
            expand(line, lambda n: n, [], {}, 1)
        elif key[0] in "rlcvik":
            # Top-level nodes cannot collide with deterministic internal namespaces.
            if any(n.lower().startswith("emi02_") for n in fields[1:3]):
                fail("unsupported_input", "reserved internal node namespace")
            if key[0] in "vi":
                match = re.fullmatch(
                    r"(\S+\s+\S+\s+\S+)\s+(PULSE|PWL)\s*\((.*)\)", line, re.I
                )
                if match:
                    numbers = match[3].split()
                    if (match[2].lower() == "pulse" and len(numbers) != 7) or (
                        match[2].lower() == "pwl"
                        and (len(numbers) < 4 or len(numbers) % 2)
                    ):
                        fail("parse_failure", "malformed reference waveform")
                    initial = number(
                        numbers[0] if match[2].lower() == "pulse" else numbers[1]
                    )
                    if match[2].lower() == "pwl" and number(numbers[0]) != 0:
                        fail("unsupported_input", "PWL must begin at time zero")
                    line = f"{match[1]} DC {initial:.17g} {match[2]}({match[3]})"
            append(line)
        else:
            fail("unsupported_input", "unsupported reference card")
    if not counters["packages"] or ".end" not in seen or ".include" not in seen:
        fail("unsupported_input", "incomplete reference circuit")
    counters["mna_unknowns"] = len(node_names) + len(branch_names)
    counters["expanded_components"] = len(native_names)
    return "\n".join(output) + "\n", counters


def import_archive(archive, deck):
    """Verify original inputs before producing caller-owned local-only model text."""
    with tempfile.TemporaryDirectory(prefix="emi02-model-") as directory:
        path = Path(directory) / "model.lib"
        try:
            provenance = adapter.adapt_archive(archive, path)
        except adapter.ModelError as exc:
            fail(exc.status, str(exc))
        result, counts = flatten(deck, path.read_bytes())
    return result, {
        "importer": VERSION,
        "ngspice_ps_vt_binding": "ambient_thermal_voltage",
        "ambient_temperature_c": REFERENCE_AMBIENT_C,
        "vt_volts": REFERENCE_THERMAL_VOLTAGE,
        **provenance,
        **counts,
        "source_deck_sha256": hashlib.sha256(deck.encode()).hexdigest(),
        "flat_deck_sha256": hashlib.sha256(result.encode()).hexdigest(),
    }
