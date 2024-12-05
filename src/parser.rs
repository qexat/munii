use std::{fmt, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Identifier,
    Integer,
    Char,
    String,
    Unit,
    Dynamic,
    Fun,
    Let,
    BracketLeft,
    BracketRight,
    ParenLeft,
    ParenRight,
    Dot,
    Comma,
    Semicolon,
    Equal,
    ColonEqual,
    Colon,
    BangEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Plus,
    Minus,
    Asterisk,
    Slash,
    Percent,
    VerticalBar,
    MinusGreater,
    Eof,
}

impl TokenKind {
    fn get_keyword(lexeme: String) -> Option<Self> {
        match lexeme.as_str() {
            "dynamic" => Some(Self::Dynamic),
            "fun" => Some(Self::Fun),
            "let" => Some(Self::Let),
            _ => None,
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
struct Token {
    kind: TokenKind,
    lexeme: String,
    position: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
struct Ast {
    body: Arc<[Stmt]>,
}

#[derive(Clone, Debug)]
enum Stmt {
    Expr(ExprStmt),
    Let(LetStmt),
    Type(TypeStmt),
}

#[derive(Clone, Debug)]
struct ExprStmt {
    expr: Expr,
}

#[derive(Clone, Debug)]
struct LetStmt {
    pattern: Pattern,
    parameters: Arc<[Parameter]>,
    return_annotation: TypeForm,
    expr: Expr,
}

#[derive(Clone, Debug)]
struct TypeStmt {
    name: TypeIdentifier,
    parameters: Arc<[TypeParameter]>,
    expr: TypeExpr,
}

#[derive(Clone, Debug)]
enum Expr {
    Identifier(IdentifierExpr),
    Literal(LiteralExpr),
    Grouping(GroupingExpr),
    Member(MemberExpr),
    App(AppExpr),
    Unary(UnaryExpr),
    Binary(BinaryExpr),
    Labelled(LabelledExpr),
    Match(MatchExpr),
    Chained(ChainedExpr),
}

#[derive(Clone, Debug)]
struct IdentifierExpr {
    lexeme: String,
}

#[derive(Clone, Debug)]
enum LiteralExpr {
    Scalar(ScalarLiteralExpr),
    Compound(CompoundLiteralExpr),
}

#[derive(Clone, Debug)]
enum ScalarLiteralExpr {
    Bool(BoolExpr),
    Integer(IntegerExpr),
    Char(CharExpr),
    String(StringExpr),
    Unit(UnitExpr),
}

#[derive(Clone, Debug)]
enum CompoundLiteralExpr {
    List(ListExpr),
}

#[derive(Clone, Debug)]
struct BoolExpr {
    lexeme: String,
}

#[derive(Clone, Debug)]
struct IntegerExpr {
    lexeme: String,
}

#[derive(Clone, Debug)]
struct CharExpr {
    lexeme: String,
}

#[derive(Clone, Debug)]
struct StringExpr {
    lexeme: String,
}

#[derive(Clone, Debug)]
struct UnitExpr;

#[derive(Clone, Debug)]
struct ListExpr {
    items: Arc<[Expr]>,
}

#[derive(Clone, Debug)]
struct GroupingExpr(Box<Expr>);

#[derive(Clone, Debug)]
struct MemberExpr {
    accessor: Box<Expr>,
    attributes: Arc<[IdentifierExpr]>,
}

#[derive(Clone, Debug)]
struct AppExpr {
    application: Box<Expr>,
    arguments: Arc<[Expr]>,
}

#[derive(Clone, Debug)]
struct UnaryExpr {
    operator: TokenKind,
    expr: Box<Expr>,
}

#[derive(Clone, Debug)]
struct BinaryExpr {
    operator: TokenKind,
    left: Box<Expr>,
    right: Box<Expr>,
}

#[derive(Clone, Debug)]
struct LabelledExpr {
    name: String,
    expr: Box<Expr>,
}

#[derive(Clone, Debug)]
struct MatchExpr {
    scrutinee: Box<Expr>,
    branches: Arc<[MatchBranch]>,
}

#[derive(Clone, Debug)]
struct MatchBranch {
    pattern: Pattern,
    leaf: Expr,
}

#[derive(Clone, Debug)]
struct ChainedExpr {
    left: Box<Expr>,
    right: Box<Expr>,
}

/* Type grammar */

#[derive(Clone, Debug)]
enum TypeForm {
    Expr(TypeExpr),
    Var(TypeVar),
}

#[derive(Clone, Debug)]
enum TypeExpr {
    // Foo Bar
    App(TypeApp),
    // Foo | Bar ; Foo -> Bar
    BinaryOp(TypeBinaryOp),
    // Foo
    Identifier(TypeIdentifier),
    // 0 ; "hello"
    Literal(TypeLiteral),
    // Foo.Bar.Baz
    Member(TypeMember),
    // Foo, Bar
    Tuple(TypeTuple),
}

#[derive(Clone, Debug)]
struct TypeApp {
    applied: Box<TypeExpr>,
    arguments: Arc<[TypeApp]>,
}

#[derive(Clone, Debug)]
struct TypeBinaryOp {
    operator: TokenKind,
    left: Box<TypeExpr>,
    right: Box<TypeExpr>,
}
#[derive(Clone, Debug)]
struct TypeIdentifier {
    lexeme: String,
}

#[derive(Clone, Debug)]
struct TypeLiteral {
    // Rust doesn't allow recursive type aliases, meaning that
    // we literally can't express constant non-scalar literals
    literal: ScalarLiteralExpr,
}

#[derive(Clone, Debug)]
struct TypeMember {
    // For now we only allow type identifiers as accessors, but
    // we might expand on it in the future - for example, we
    // could allow type applications
    accessor: TypeIdentifier,
    attributes: Arc<[TypeIdentifier]>,
}

#[derive(Clone, Debug)]
struct TypeTuple {
    items: Arc<[TypeExpr]>,
}

#[derive(Clone, Debug)]
struct TypeVar {
    // Deliberate choice of not using de Bruijn indices ;)
    name: String,
}
