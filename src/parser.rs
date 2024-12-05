use std::fmt;

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
struct Ast<'a> {
    body: Vec<Stmt<'a>>,
}

#[derive(Clone, Debug)]
enum Stmt<'a> {
    Expr(ExprStmt<'a>),
    Let(LetStmt<'a>),
    Type(TypeStmt<'a>),
}

#[derive(Clone, Debug)]
struct ExprStmt<'a> {
    expr: Expr<'a>,
}

#[derive(Clone, Debug)]
struct LetStmt<'a> {
    pattern: Pattern<'a>,
    parameters: &'a [Parameter<'a>],
    return_annotation: TypeForm<'a>,
    expr: Expr<'a>,
}

#[derive(Clone, Debug)]
struct TypeStmt<'a> {
    name: String,
    parameters: &'a [TypeParameter<'a>],
    expr: TypeExpr<'a>,
}

#[derive(Clone, Debug)]
enum Expr<'a> {
    Identifier(IdentifierExpr),
    Bool(BoolExpr),
    Integer(IntegerExpr),
    Char(CharExpr),
    String(StringExpr),
    Unit(UnitExpr),
    List(ListExpr<'a>),
    Grouping(GroupingExpr<'a>),
    Member(MemberExpr<'a>),
    App(AppExpr<'a>),
    Unary(UnaryExpr<'a>),
    Binary(BinaryExpr<'a>),
    Labelled(LabelledExpr<'a>),
    Match(MatchExpr<'a>),
    Chained(ChainedExpr<'a>),
}

#[derive(Clone, Debug)]
struct IdentifierExpr {
    lexeme: String,
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
struct ListExpr<'a> {
    items: &'a [Expr<'a>],
}

#[derive(Clone, Debug)]
struct GroupingExpr<'a>(Box<Expr<'a>>);

#[derive(Clone, Debug)]
struct MemberExpr<'a> {
    accessor: Box<Expr<'a>>,
    attribute_list: &'a [String],
}

#[derive(Clone, Debug)]
struct AppExpr<'a> {
    application: Box<Expr<'a>>,
    arguments: &'a [Expr<'a>],
}

#[derive(Clone, Debug)]
struct UnaryExpr<'a> {
    operator: TokenKind,
    expr: Box<Expr<'a>>,
}

#[derive(Clone, Debug)]
struct BinaryExpr<'a> {
    operator: TokenKind,
    left: Box<Expr<'a>>,
    right: Box<Expr<'a>>,
}

#[derive(Clone, Debug)]
struct LabelledExpr<'a> {
    name: String,
    expr: Box<Expr<'a>>,
}

#[derive(Clone, Debug)]
struct MatchExpr<'a> {
    scrutinee: Box<Expr<'a>>,
    branches: &'a [MatchBranch<'a>],
}

#[derive(Clone, Debug)]
struct MatchBranch<'a> {
    pattern: Pattern<'a>,
    leaf: Expr<'a>,
}

#[derive(Clone, Debug)]
struct ChainedExpr<'a> {
    left: Box<Expr<'a>>,
    right: Box<Expr<'a>>,
}
