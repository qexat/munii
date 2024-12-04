# ruff: noqa: PGH004
# ruff: noqa
"""
munii
"""

from __future__ import annotations

import abc
import enum
import sys
import typing

import attrs
import option
import result

if typing.TYPE_CHECKING:
    import collections.abc

type _ = typing.Any


class TokenKind(enum.Enum):
    IDENTIFIER = enum.auto()
    INTEGER = enum.auto()
    STRING = enum.auto()
    UNIT = enum.auto()

    # *- Keywords -* #
    DYNAMIC = enum.auto()
    FUN = enum.auto()
    LET = enum.auto()

    BRACKET_LEFT = enum.auto()
    BRACKET_RIGHT = enum.auto()
    PAREN_LEFT = enum.auto()
    PAREN_RIGHT = enum.auto()

    DOT = enum.auto()
    COMMA = enum.auto()
    SEMICOLON = enum.auto()

    EQUAL = enum.auto()
    COLON_EQUAL = enum.auto()
    COLON = enum.auto()

    BANG_EQUAL = enum.auto()
    GREATER = enum.auto()
    GREATER_EQUAL = enum.auto()
    LESS = enum.auto()
    LESS_EQUAL = enum.auto()

    PLUS = enum.auto()
    MINUS = enum.auto()
    ASTERISK = enum.auto()
    SLASH = enum.auto()
    PERCENT = enum.auto()

    VERTICAL_BAR = enum.auto()
    MINUS_GREATER = enum.auto()

    EOF = enum.auto()


LEXEME_TO_KEYWORD_MAPPING = {
    "dynamic": TokenKind.DYNAMIC,
    "fun": TokenKind.FUN,
    "let": TokenKind.LET,
}


class Token(typing.NamedTuple):
    kind: TokenKind
    lexeme: str
    position: int


@attrs.frozen
class Ast:
    body: list[Stmt]


class StmtVisitor[R_co](typing.Protocol):
    def visit_expr_stmt(self, stmt: ExprStmt) -> R_co: ...


@attrs.frozen
class StmtBase(abc.ABC):
    @abc.abstractmethod
    def accept[R](self, visitor: StmtVisitor[R]) -> R:
        pass


@attrs.frozen
class ExprStmt(StmtBase):
    expr: Expr

    @typing.override
    def accept[R](self, visitor: StmtVisitor[R]) -> R:
        return visitor.visit_expr_stmt(self)


type Stmt = ExprStmt


class ExprVisitor[R_co](typing.Protocol):
    # *- prim_expr -* #
    def visit_ident_expr(self, expr: IdentExpr) -> R_co: ...
    def visit_integer_expr(self, expr: IntegerExpr) -> R_co: ...
    def visit_char_expr(self, expr: CharExpr) -> R_co: ...
    def visit_string_expr(self, expr: StringExpr) -> R_co: ...

    # *- grouping_expr -* #
    def visit_grouping_expr(self, expr: GroupingExpr) -> R_co: ...

    # *- binary_expr -* #
    def visit_binary_expr(self, expr: BinaryExpr) -> R_co: ...


@attrs.frozen
class ExprBase(abc.ABC):
    ty: Type

    @abc.abstractmethod
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        pass


@attrs.frozen
class IdentExpr(ExprBase):
    identifier: str

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_ident_expr(self)


@attrs.frozen
class IntegerExpr(ExprBase):
    token: Token

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_integer_expr(self)


@attrs.frozen
class CharExpr(ExprBase):
    token: Token

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_char_expr(self)


@attrs.frozen
class StringExpr(ExprBase):
    token: Token

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_string_expr(self)


type PrimExpr = IdentExpr | IntegerExpr | CharExpr | StringExpr


@attrs.frozen
class GroupingExpr(ExprBase):
    grouped: Expr

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_grouping_expr(self)


@attrs.frozen
class BinaryExpr(ExprBase):
    operator: TokenKind
    left: Expr
    right: Expr

    @typing.override
    def accept[R](self, visitor: ExprVisitor[R]) -> R:
        return visitor.visit_binary_expr(self)


type Expr = PrimExpr | GroupingExpr | BinaryExpr


class TypeVisitor[R_co](typing.Protocol):
    def visit_primitive_type(self, ty: PrimitiveType) -> R_co: ...


@attrs.frozen
class TypeBase(abc.ABC):
    @abc.abstractmethod
    def accept[R](self, visitor: TypeVisitor[R]) -> R:
        pass


@attrs.frozen
class PrimitiveType(TypeBase):
    variant: PrimitiveTypeVariant

    @typing.override
    def accept[R](self, visitor: TypeVisitor[R]) -> R:
        return visitor.visit_primitive_type(self)


type Type = PrimitiveType


class PrimitiveTypeVariant(enum.Enum):
    BOOL = enum.auto()
    INT = enum.auto()
    NAT = enum.auto()
    CHAR = enum.auto()
    STRING = enum.auto()
    UNIT = enum.auto()


class MuniiErrorVisitor[R_co](typing.Protocol):
    def visit_invalid_character_literal_error(
        self,
        error: InvalidCharacterLiteralError,
    ) -> R_co: ...
    def visit_unrecognized_token_error(
        self,
        error: UnrecognizedTokenError,
    ) -> R_co: ...
    def visit_unterminated_string_literal_error(
        self,
        error: UnterminatedStringLiteralError,
    ) -> R_co: ...


@attrs.frozen
class MuniiErrorBase(abc.ABC):
    context: str
    position: int

    @abc.abstractmethod
    def accept[R](self, visitor: MuniiErrorVisitor[R]) -> R:
        pass


@attrs.frozen
class InvalidCharacterLiteralError(MuniiErrorBase):
    token: Token
    reason: str

    @typing.override
    def accept[R](self, visitor: MuniiErrorVisitor[R]) -> R:
        return visitor.visit_invalid_character_literal_error(self)


@attrs.frozen
class UnrecognizedTokenError(MuniiErrorBase):
    lexeme: str
    potential_matches: list[str] = attrs.field(factory=list)

    @typing.override
    def accept[R](self, visitor: MuniiErrorVisitor[R]) -> R:
        return visitor.visit_unrecognized_token_error(self)


@attrs.frozen
class UnterminatedStringLiteralError(MuniiErrorBase):
    @typing.override
    def accept[R](self, visitor: MuniiErrorVisitor[R]) -> R:
        return visitor.visit_unterminated_string_literal_error(self)


type Error = (
    InvalidCharacterLiteralError
    | UnrecognizedTokenError
    | UnterminatedStringLiteralError
)


@attrs.frozen
class RuntimeResult:
    ast: Ast
    value: option.Option[_]


@attrs.frozen
class Beqona:
    def get_input_loop(self) -> collections.abc.Generator[str, None, None]:
        try:
            while True:
                prompt = "\x1b[1;34m>>>\x1b[22;39m "
                try:
                    yield input(prompt)
                except KeyboardInterrupt:
                    print()
                    continue
        except EOFError:
            return

    def lex(self, source: str) -> result.Result[list[Token], Error]:
        """
        Scan the source to find tokens.
        """

        return Lexer(source).run()

    def parse(self, tokens: list[Token]) -> result.Result[Ast, Error]:
        """
        Parse the sequence of tokens into an abstract syntax
        tree.
        """
        ...

    def analyze(self, ast: Ast) -> result.Result[Ast, Error]:
        """
        Analyze semantically the abstract syntax tree to find
        issues such as nonexisting names.
        """
        ...

    def typecheck(self, ast: Ast) -> result.Result[Ast, Error]:
        """
        Infer and check the types of the abstract syntax
        nodes.
        """
        ...

    def evaluate(self, ast: Ast) -> RuntimeResult:
        """
        Evaluate the abstract syntax tree.
        """
        ...

    def refine(self, ast: Ast) -> result.Result[Ast, Error]:
        """
        Refine the type inference using runtime feedback.
        """
        ...

    def main(self) -> int:
        for line in self.get_input_loop():
            if not line:
                print(end="\x1b[A")
                continue

            output_prefix = "\x1b[1;35m<<<\x1b[22;39m"

            match self.lex(line):
                case result.Err(error):
                    print(f"Error: {error}", file=sys.stderr)
                case result.Ok(tokens):
                    for token in tokens:
                        print(f"{output_prefix} {token}")

        return 0


@attrs.define
class Lexer:
    # *- Parameters -* #
    source: typing.Final[str]
    # *- State -* #
    start: int = attrs.field(init=False, default=0)
    current: int = attrs.field(init=False, default=0)
    tokens: list[Token] = attrs.field(init=False, factory=list)

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def synchronize_start(self) -> None:
        self.start = self.current

    def get_current_lexeme(self) -> str:
        return self.source[self.start : self.current]

    def build_token(self, kind: TokenKind) -> Token:
        return Token(kind, self.get_current_lexeme(), self.start)

    def peek(self) -> str:
        if self.is_at_end():
            return "\0"
        return self.source[self.current]

    def advance(self) -> None:
        self.current += 1

    def consume(self) -> str:
        char = self.peek()
        self.advance()

        return char

    def satisfies_predicate(
        self,
        predicate: collections.abc.Callable[[str], bool],
    ) -> bool:
        if predicate(self.peek()):
            self.advance()
            return True

        return False

    def matches(self, char: str) -> bool:
        return self.satisfies_predicate(lambda current: current == char)

    def matches_one_of(self, *chars: str) -> bool:
        return self.satisfies_predicate(lambda current: current in chars)

    def not_matches(self, char: str) -> bool:
        return self.satisfies_predicate(lambda current: current != char)

    def matches_none_of(self, *chars: str) -> bool:
        return self.satisfies_predicate(lambda current: current not in chars)

    def scan_string(self) -> result.Result[TokenKind, Error]:
        while self.matches_none_of('"', "\n", "\0"):
            continue

        if not self.matches('"'):
            return result.Err(
                UnterminatedStringLiteralError(
                    self.source,
                    self.start,
                )
            )

        return result.Ok(TokenKind.STRING)

    def scan_integer(self) -> result.Result[TokenKind, Error]:
        while self.satisfies_predicate(str.isdecimal):
            continue

        return result.Ok(TokenKind.INTEGER)

    def scan_keyword_or_identifier(self) -> TokenKind:
        while self.satisfies_predicate(str.isidentifier):
            continue

        lexeme = self.get_current_lexeme()

        return LEXEME_TO_KEYWORD_MAPPING.get(lexeme, TokenKind.IDENTIFIER)

    def scan_token(self) -> result.Result[Token, Error]:
        token_kind: TokenKind | None = None

        while token_kind is None:
            match self.consume():
                case "\0":
                    token_kind = TokenKind.EOF
                case " " | "\r" | "\t" | "\n":
                    self.synchronize_start()
                    continue
                case ":" if self.matches("="):
                    token_kind = TokenKind.COLON_EQUAL
                case "!" if self.matches("="):
                    token_kind = TokenKind.BANG_EQUAL
                case ">" if self.matches("="):
                    token_kind = TokenKind.GREATER_EQUAL
                case "<" if self.matches("="):
                    token_kind = TokenKind.LESS_EQUAL
                case "-" if self.matches(">"):
                    token_kind = TokenKind.MINUS_GREATER
                case "(" if self.matches(")"):
                    token_kind = TokenKind.UNIT
                case "[":
                    token_kind = TokenKind.BRACKET_LEFT
                case "]":
                    token_kind = TokenKind.BRACKET_RIGHT
                case "(":
                    token_kind = TokenKind.PAREN_LEFT
                case ")":
                    token_kind = TokenKind.PAREN_RIGHT
                case ".":
                    token_kind = TokenKind.DOT
                case ",":
                    token_kind = TokenKind.COMMA
                case ";":
                    token_kind = TokenKind.SEMICOLON
                case ":":
                    token_kind = TokenKind.COLON
                case "=":
                    token_kind = TokenKind.EQUAL
                case ">":
                    token_kind = TokenKind.GREATER
                case "<":
                    token_kind = TokenKind.LESS
                case "+":
                    token_kind = TokenKind.PLUS
                case "-":
                    token_kind = TokenKind.MINUS
                case "*":
                    token_kind = TokenKind.ASTERISK
                case "/":
                    token_kind = TokenKind.SLASH
                case "%":
                    token_kind = TokenKind.PERCENT
                case "|":
                    token_kind = TokenKind.VERTICAL_BAR
                case '"':
                    token_kind_result = self.scan_string()

                    if isinstance(token_kind_result, result.Err):
                        return token_kind_result

                    token_kind = token_kind_result.unwrap()
                case char if char.isdecimal():
                    token_kind_result = self.scan_integer()

                    if isinstance(token_kind_result, result.Err):
                        return token_kind_result

                    token_kind = token_kind_result.unwrap()
                case char if char.isidentifier():
                    token_kind = self.scan_keyword_or_identifier()
                case char:
                    return result.Err(
                        UnrecognizedTokenError(
                            self.source,
                            self.start,
                            self.get_current_lexeme(),
                        )
                    )

        return result.Ok(self.build_token(token_kind))

    def run(self) -> result.Result[list[Token], Error]:
        while not self.is_at_end():
            self.synchronize_start()
            token_result = self.scan_token()

            if isinstance(token_result, result.Err):
                return token_result

            self.tokens.append(token_result.unwrap())

        self.synchronize_start()
        self.tokens.append(self.build_token(TokenKind.EOF))

        return result.Ok(self.tokens)


class BacktrackResult(enum.Enum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()


@attrs.define
class Parser:
    # *- Parameters -* #
    tokens: typing.Final[list[Token]]
    # *- State -* #
    start_stack: list[int] = attrs.field(init=False, factory=list)
    current: int = attrs.field(init=False, default=0)
    unrecoverable: bool = attrs.field(init=False, default=False)

    def is_at_end(self) -> bool:
        return self.current >= len(self.tokens)

    def start_rule(self) -> None:
        self.start_stack.append(self.current)

    def abort_rule[E: Error](self, error: result.Err[E]) -> result.Err[E]:
        self.backtrack()

        return error

    def backtrack(self) -> None:
        if self.start_stack:
            self.current = self.start_stack.pop()
        else:
            self.unrecoverable = True

    def peek(self) -> Token:
        if self.is_at_end():
            return Token(TokenKind.EOF, "\0", -1)

        return self.tokens[self.current]

    def advance(self) -> None:
        self.current += 1

    def consume(self) -> Token:
        token = self.peek()
        self.advance()

        return token

    def parse_expr(self) -> result.Result[Expr, Error]:
        self.start_rule()

    def parse_expr_stmt(self) -> result.Result[ExprStmt, Error]:
        self.start_rule()

        expr = self.parse_expr()

        if isinstance(expr, result.Err):
            return self.abort_rule(expr)

        return result.Ok(ExprStmt(expr.unwrap()))

    def parse_stmt(self) -> result.Result[Stmt, Error]:
        return self.parse_expr_stmt()

    def parse_ast(self) -> result.Result[Ast, Error]:
        body: list[Stmt] = []

        while not self.is_at_end():
            stmt_result = self.parse_stmt()

            if isinstance(stmt_result, result.Err):
                return stmt_result

            body.append(stmt_result.unwrap())

        return result.Ok(Ast(body))

    def run(self) -> result.Result[Ast, Error]:
        return self.parse_ast()


def main() -> int:
    """
    Main function.
    """

    return Beqona().main()


if __name__ == "__main__":
    raise SystemExit(main())
