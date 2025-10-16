from __future__ import annotations

import argparse
import itertools
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Load input/output tokens
from .transformer_classifier import load_tokens_and_labels_from_token_py


# ----- AST definitions -----
@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Const:
    value: bool  # True for ⊤, False for ⊥


@dataclass(frozen=True)
class Not:
    expr: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str  # one of "∧", "∨", "→"
    left: "Expr"
    right: "Expr"


Expr = Var | Const | Not | BinOp


def expr_to_string(e: Expr) -> str:
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Const):
        return "⊤" if e.value else "⊥"
    if isinstance(e, Not):
        s = expr_to_string(e.expr)
        # Keep parentheses around non-atomic to avoid ambiguity
        if isinstance(e.expr, (BinOp)):
            return f"¬({s})"
        return f"¬{s}"
    if isinstance(e, BinOp):
        ls = expr_to_string(e.left)
        rs = expr_to_string(e.right)
        return f"({ls} {e.op} {rs})"
    raise TypeError("unknown expr type")


def evaluate(e: Expr, env: Dict[str, bool]) -> bool:
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Not):
        return not evaluate(e.expr, env)
    if isinstance(e, BinOp):
        l = evaluate(e.left, env)
        r = evaluate(e.right, env)
        if e.op == "∧":
            return l and r
        if e.op == "∨":
            return l or r
        if e.op == "→":
            return (not l) or r
        raise ValueError(f"unknown op {e.op}")
    raise TypeError("unknown expr type")


def is_tautology(e: Expr, variables: List[str]) -> bool:
    for values in itertools.product([False, True], repeat=len(variables)):
        env = {v: val for v, val in zip(variables, values)}
        if not evaluate(e, env):
            return False
    return True


class FormulaGenerator:
    def __init__(
        self,
        variables: List[str],
        allow_const: bool = True,
        difficulty: float = 0.5,
        max_depth: int = 4,
        binary_op_weights: Optional[Dict[str, float]] = None,
        unary_weight: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.variables = variables
        self.allow_const = allow_const
        self.difficulty = max(0.0, min(1.0, difficulty))
        # Scale depth by difficulty (harder => deeper)
        self.max_depth = max(1, int(round(max_depth * (0.5 + self.difficulty))))
        self.unary_weight = max(0.0, min(1.0, unary_weight))
        self.rng = random.Random(seed)

        self.binary_ops = ["∧", "∨", "→"]
        if binary_op_weights is None:
            self.binary_op_weights = {"∧": 1.0, "∨": 1.0, "→": 1.0}
        else:
            self.binary_op_weights = {op: float(binary_op_weights.get(op, 1.0)) for op in self.binary_ops}

    def _choice(self, items: List[str], weights: Optional[List[float]] = None) -> str:
        if weights is None:
            return self.rng.choice(items)
        total = sum(weights)
        x = self.rng.random() * total
        c = 0.0
        for item, w in zip(items, weights):
            c += w
            if x <= c:
                return item
        return items[-1]

    def _gen_atom(self) -> Expr:
        atoms: List[Expr] = [Var(v) for v in self.variables]
        if self.allow_const:
            atoms.append(Const(True))
            atoms.append(Const(False))
        return self.rng.choice(atoms)

    def _maybe_negate(self, e: Expr) -> Expr:
        # More negation as difficulty grows
        if self.rng.random() < (self.unary_weight * (0.5 + 0.5 * self.difficulty)):
            return Not(e)
        return e

    def _gen_rec(self, depth: int) -> Expr:
        if depth <= 0 or self.rng.random() < (0.25 - 0.2 * (1 - self.difficulty)):
            return self._maybe_negate(self._gen_atom())

        op = self._choice(
            self.binary_ops,
            [self.binary_op_weights[op] for op in self.binary_ops],
        )
        left = self._gen_rec(depth - 1)
        right = self._gen_rec(depth - 1)
        node: Expr = BinOp(op=op, left=left, right=right)
        return self._maybe_negate(node)

    def generate(self) -> Expr:
        return self._gen_rec(self.max_depth)


def filter_formulas(
    gen: FormulaGenerator,
    max_len: int,
    require_tautology: bool,
    limit: int,
) -> List[str]:
    out: List[str] = []
    variables = gen.variables
    attempts = 0
    # Keep a cap on attempts to avoid infinite loops
    max_attempts = limit * 500
    while len(out) < limit and attempts < max_attempts:
        attempts += 1
        e = gen.generate()
        s = expr_to_string(e)
        if len(s) > max_len:
            continue
        if require_tautology and not is_tautology(e, variables):
            continue
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random propositional formulas")
    parser.add_argument("--count", type=int, default=10, help="number of formulas to output")
    parser.add_argument("--max_len", type=int, default=50, help="max string length per formula")
    parser.add_argument("--difficulty", type=float, default=0.5, help="0.0-1.0 depth/negation difficulty")
    parser.add_argument("--allow_const", action="store_true", help="allow ⊤ and ⊥ in atoms")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tautology_only", action="store_true", help="keep only tautologies")
    args = parser.parse_args()

    token_py_path = os.path.join(os.path.dirname(__file__), "fof_tokens.py")
    input_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)

    # Infer variable set from available tokens (a, b, c)
    variables = [t for t in ["a", "b", "c"] if t in input_tokens]
    if not variables:
        variables = ["a", "b", "c"]

    gen = FormulaGenerator(
        variables=variables,
        allow_const=args.allow_const,
        difficulty=args.difficulty,
        max_depth=4,
        seed=args.seed,
    )

    formulas = filter_formulas(
        gen=gen,
        max_len=args.max_len,
        require_tautology=args.tautology_only,
        limit=args.count,
    )

    for s in formulas:
        print(s)


if __name__ == "__main__":
    main()


