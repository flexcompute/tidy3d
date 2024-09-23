import importlib
import inspect
import re
from enum import Enum, auto
from typing import List, Tuple

from .functions import Function
from .operators import Abs
from .types import ExpressionType
from .variables import Constant, Variable

# Dynamically collect all function classes
function_classes = {}
functions_module = importlib.import_module(".functions", package=__package__)
for name, obj in inspect.getmembers(functions_module):
    if inspect.isclass(obj) and issubclass(obj, Function) and obj is not Function:
        function_classes[name.lower()] = obj

function_names_pattern = r"|".join(map(re.escape, function_classes.keys()))


class ParserError(Exception):
    """Custom exception class for parser errors."""


class TokenType(Enum):
    NUMBER = auto()
    FUNC = auto()
    IDENT = auto()
    OP = auto()
    ABS = auto()
    SKIP = auto()
    MISMATCH = auto()


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
        # Collect all function names including 'abs' (since it's a function in expressions)
        function_names = set(function_classes.keys())  # Convert to a set for faster lookup
        function_names.add("abs")

        # Insert '*' between a number and a variable/function/parenthesis, excluding 'e' or 'E'
        s = re.sub(r"(\d)(?![eE])(?=\s*[A-Za-z_(|])", r"\1*", s)

        # Insert '*' between a closing parenthesis and an opening parenthesis
        s = re.sub(r"(\))(?=\s*\()", r"\1*", s)

        # Insert '*' between a closing parenthesis and a variable/function/parenthesis
        s = re.sub(r"(\))(?=\s*[A-Za-z_(|])", r"\1*", s)

        # Insert '*' between an identifier and a number
        s = re.sub(r"([A-Za-z_]\w*)(?=\s*\d)", r"\1*", s)

        # Insert '*' between an identifier and '(', if identifier is not a function
        def replace_if_not_function(match):
            identifier = match.group(1)
            if identifier.lower() not in function_names:
                return identifier + "*"
            else:
                return identifier

        s = re.sub(r"([A-Za-z_]\w*)(?=\s*\()", replace_if_not_function, s)

        return s

    def tokenize(self, s: str) -> List[Tuple[TokenType, str]]:
        """Convert the input string into a list of tokens."""
        token_specification = [
            (
                TokenType.NUMBER,
                r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            ),  # Match numbers, including scientific notation
            (TokenType.FUNC, rf"\b(?:{function_names_pattern})\b"),  # Functions
            (TokenType.IDENT, r"[A-Za-z_]\w*"),  # Identifiers
            (TokenType.OP, r"\*\*|//|%|@|[+\-*/^()]"),  # Operators
            (TokenType.ABS, r"\|"),  # Absolute value bars
            (TokenType.SKIP, r"\s+"),  # Skip spaces and tabs
            (TokenType.MISMATCH, r"."),  # Any other character
        ]
        token_regex = "|".join(f"(?P<{pair[0].name}>{pair[1]})" for pair in token_specification)
        get_token = re.compile(token_regex, re.IGNORECASE).match
        pos = 0
        tokens = []
        while pos < len(s):
            match = get_token(s, pos)
            if match is not None:
                kind = TokenType[match.lastgroup]
                value = match.group()
                if kind == TokenType.NUMBER:
                    tokens.append((kind, value))
                elif kind == TokenType.FUNC:
                    tokens.append((kind, value.lower()))
                elif kind == TokenType.IDENT:
                    tokens.append((kind, value))
                elif kind == TokenType.OP:
                    tokens.append((kind, value))
                elif kind == TokenType.ABS:
                    tokens.append((kind, value))
                elif kind == TokenType.SKIP:
                    pass  # Ignore whitespace
                elif kind == TokenType.MISMATCH:
                    raise ParserError(f"Unexpected character {value!r} at position {pos}")
                pos = match.end()
            else:
                raise ParserError(f"Unexpected character {s[pos]!r} at position {pos}")
        return tokens

    def peek(self) -> Tuple[TokenType, str]:
        """Peek at the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        else:
            return (TokenType.MISMATCH, "")

    def advance(self) -> Tuple[TokenType, str]:
        """Consume the current token and advance to the next one."""
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, expected_type: TokenType, expected_value: str = None):
        """Assert that the current token matches the expected type and value."""
        token_type, token_value = self.peek()
        if token_type != expected_type or (expected_value and token_value != expected_value):
            raise ParserError(
                f"Expected {expected_type.name} {expected_value}, but got {token_type.name} {token_value}"
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
        """Parse a term (handles multiplication, division, floor division, modulus)."""
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
            elif token_value == "//":
                self.advance()
                right = self.factor()
                expr = expr // right
            elif token_value == "%":
                self.advance()
                right = self.factor()
                expr = expr % right
            elif token_value == "@":
                self.advance()
                right = self.term()
                expr = expr @ right
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
        if token_type == TokenType.ABS:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.ABS)
            return Abs(operand=expr)
        else:
            return self.primary()

    def primary(self) -> ExpressionType:
        """Parse a primary expression (numbers, variables, functions, parentheses)."""
        token_type, token_value = self.peek()
        if token_value == "(":
            self.advance()
            expr = self.expression()
            self.expect(TokenType.OP, ")")
            return expr
        elif token_type == TokenType.NUMBER:
            self.advance()
            value = (
                float(token_value)
                if "." in token_value or "e" in token_value.lower()
                else int(token_value)
            )
            return Constant(value)
        elif token_type == TokenType.IDENT:
            self.advance()
            return Variable(name=token_value)
        elif token_type == TokenType.FUNC:
            self.advance()
            func_name = token_value.lower()
            self.expect(TokenType.OP, "(")
            arg = self.expression()
            self.expect(TokenType.OP, ")")
            if func_name in function_classes:
                func_class = function_classes[func_name]
                return func_class(operand=arg)
            elif func_name == "abs":
                # Handle the special case for 'abs' if necessary
                return Abs(operand=arg)
            else:
                raise ParserError(f"Unknown function {func_name}")
        else:
            raise ParserError(f"Unexpected token {token_type.name} {token_value}")
