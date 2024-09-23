import re
from typing import List, Tuple

from .functions import Cos, Exp, Log, Log10, Sin, Tan
from .operators import Abs
from .types import ExpressionType
from .variables import Constant, Variable


class ParserError(Exception):
    """Custom exception class for parser errors."""


class Parser:
    """
    A parser to convert mathematical expressions in string form
    into serializable ExpressionType objects defined in the expressions module.
    """

    def parse(self, expression: str) -> ExpressionType:
        """Parse the input expression string and return an ExpressionType object."""
        # Preprocess to handle implicit multiplication
        expression = self.insert_implicit_multiplication(expression)
        self.tokens = self.tokenize(expression)
        self.pos = 0
        result = self.expression()
        if self.pos < len(self.tokens):
            raise ParserError(f"Unexpected token at end of expression: {self.peek()}")
        return result

    def insert_implicit_multiplication(self, s: str) -> str:
        """
        Inserts explicit multiplication operators where multiplication is implied,
        such as between a number and a variable, between a variable and a parenthesis (if there is space),
        or between a closing parenthesis and an opening parenthesis.
        """
        # Define function names to exclude from insertion
        function_names = ["sin", "cos", "tan", "exp", "log", "log10", "sqrt", "abs"]
        function_pattern = r"|".join(function_names)

        # Insert '*' between a number and a variable/function/parenthesis
        s = re.sub(r"(\d)\s*(?=[A-Za-z_(|])", r"\1*", s)

        # Insert '*' between a closing parenthesis and an opening parenthesis
        s = re.sub(r"(\))\s*(\()", r"\1*\2", s)

        # Insert '*' between a closing parenthesis and a variable/function/parenthesis
        s = re.sub(r"(\))\s*([A-Za-z_(|])", r"\1*\2", s)

        # Insert '*' between an identifier and a number
        s = re.sub(r"([A-Za-z_])\s*(\d)", r"\1*\2", s)

        # Insert '*' between an identifier and '(', only if there's space and identifier is not a function
        pattern = r"\b(?!(" + function_pattern + r")\b)([A-Za-z_]\w*)\s+(\()"
        s = re.sub(pattern, r"\2*\3", s)

        return s

    def tokenize(self, s: str) -> List[Tuple[str, str]]:
        """Convert the input string into a list of tokens."""
        token_specification = [
            ("NUMBER", r"\d+(\.\d*)?"),  # Integer or decimal number
            ("FUNC", r"\b(?:abs|sin|cos|tan|exp|log10?|sqrt)\b"),  # Functions
            ("IDENT", r"[A-Za-z_]\w*"),  # Identifiers
            ("OP", r"\*\*|[\+\-\*/\^\(\)]"),  # Operators and parentheses
            ("ABS", r"\|"),  # Absolute value bars
            ("SKIP", r"\s+"),  # Skip spaces and tabs
            ("MISMATCH", r"."),  # Any other character
        ]
        token_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
        get_token = re.compile(token_regex, re.IGNORECASE).match
        pos = 0
        tokens = []
        match = get_token(s, pos)
        while match is not None:
            kind = match.lastgroup
            value = match.group()
            if kind == "NUMBER":
                tokens.append(("NUMBER", value))
            elif kind == "FUNC":
                tokens.append(("FUNC", value.lower()))
            elif kind == "IDENT":
                tokens.append(("IDENT", value))
            elif kind == "OP":
                tokens.append(("OP", value))
            elif kind == "ABS":
                tokens.append(("ABS", value))
            elif kind == "SKIP":
                pass  # Ignore whitespace
            elif kind == "MISMATCH":
                raise ParserError(f"Unexpected character {value!r} at position {pos}")
            pos = match.end()
            match = get_token(s, pos)
        return tokens

    def peek(self) -> Tuple[str, str]:
        """Peek at the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        else:
            return ("EOF", "")

    def advance(self) -> Tuple[str, str]:
        """Consume the current token and advance to the next one."""
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, expected_type: str, expected_value: str = None):
        """Assert that the current token matches the expected type and value."""
        token_type, token_value = self.peek()
        if token_type != expected_type or (expected_value and token_value != expected_value):
            raise ParserError(
                f"Expected {expected_type} {expected_value}, but got {token_type} {token_value}"
            )
        self.advance()

    def expression(self) -> ExpressionType:
        """Parse an expression (handles addition and subtraction)."""
        expr = self.term()
        while True:
            token_type, token_value = self.peek()
            if token_value == "+":
                self.advance()
                right = self.term()
                expr = expr + right
            elif token_value == "-":
                self.advance()
                right = self.term()
                expr = expr - right
            else:
                break
        return expr

    def term(self) -> ExpressionType:
        """Parse a term (handles multiplication and division)."""
        expr = self.factor()
        while True:
            token_type, token_value = self.peek()
            if token_value == "*":
                self.advance()
                right = self.factor()
                expr = expr * right
            elif token_value == "/":
                self.advance()
                right = self.factor()
                expr = expr / right
            else:
                break
        return expr

    def factor(self) -> ExpressionType:
        """Parse a factor (handles exponents)."""
        expr = self.unary()
        while True:
            token_type, token_value = self.peek()
            if token_value in ("^", "**"):
                self.advance()
                right = self.unary()
                expr = expr**right
            else:
                break
        return expr

    def unary(self) -> ExpressionType:
        """Parse a unary expression (handles negation)."""
        token_type, token_value = self.peek()
        if token_value == "+":
            self.advance()
            return self.unary()
        elif token_value == "-":
            self.advance()
            expr = self.unary()
            return -expr
        else:
            return self.abs_expression()

    def abs_expression(self) -> ExpressionType:
        """Parse expressions that may start with an absolute value bar."""
        token_type, token_value = self.peek()
        if token_type == "ABS":
            self.advance()
            expr = self.expression()
            self.expect("ABS")
            return Abs(operand=expr)
        else:
            return self.primary()

    def primary(self) -> ExpressionType:
        """Parse a primary expression (numbers, variables, functions, parentheses)."""
        token_type, token_value = self.peek()
        if token_value == "(":
            self.advance()
            expr = self.expression()
            self.expect("OP", ")")
            return expr
        elif token_type == "NUMBER":
            self.advance()
            value = float(token_value) if "." in token_value else int(token_value)
            return Constant(value)
        elif token_type == "IDENT":
            self.advance()
            return Variable(name=token_value)
        elif token_type == "FUNC":
            self.advance()
            func_name = token_value
            self.expect("OP", "(")
            arg = self.expression()
            self.expect("OP", ")")
            func_map = {
                "sin": Sin,
                "cos": Cos,
                "tan": Tan,
                "exp": Exp,
                "log": Log,
                "log10": Log10,
                "sqrt": lambda x: x ** Constant(0.5),
                "abs": Abs,
            }
            if func_name in func_map:
                func_class = func_map[func_name]
                if func_name == "abs":
                    return func_class(operand=arg)
                else:
                    return func_class(arg)
            else:
                raise ParserError(f"Unknown function {func_name}")
        else:
            raise ParserError(f"Unexpected token {token_type} {token_value}")
