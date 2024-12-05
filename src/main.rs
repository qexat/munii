mod parser;

use std::{fmt, usize};

trait IsMuniiIdentifier {
    fn is_identifier(self) -> bool;
}

impl IsMuniiIdentifier for char {
    fn is_identifier(self) -> bool {
        match self {
            'A'..='Z' | 'a'..='z' | '_' => true,
            _ => false,
        }
    }
}

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
    body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
enum Stmt {
    ExprStmt { expr: Expr },
}

#[derive(Clone, Debug)]
struct TypedExpr {
    expr: Expr,
    static_type: Type,
    runtime_type: Option<Type>,
}

#[derive(Clone, Debug)]
enum Expr {
    IdentExpr {
        identifier: String,
    },
    IntegerExpr {
        token: Token,
    },
    CharExpr {
        token: Token,
    },
    StringExpr {
        token: Token,
    },
    GroupingExpr {
        grouped: Box<Expr>,
    },
    BinaryExpr {
        operator: TokenKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}
#[derive(Clone, Debug)]
enum Type {
    PrimitiveType(PrimitiveVariant),
}

#[derive(Clone, Debug)]
enum PrimitiveVariant {
    Bool,
    Int,
    Nat,
    Char,
    String,
    Unit,
}

#[derive(Clone, Debug)]

enum ErrKind {
    ExpectedExpression,
    InvalidCharacterLiteral {
        reason: &'static str,
    },
    UnimplementedExprRule {
        precedence: Precedence,
    },
    UnmatchedParenthesis {
        left_paren: Token,
    },
    UnrecognizedToken {
        token: char,
        potential_matches: Vec<String>,
    },
    UnterminatedStringLiteral,
}

#[derive(Clone, Debug)]
struct Error {
    kind: ErrKind,
    position: usize,
}

trait Analyzer<I, T: Eq> {
    fn is_at_end(&self) -> bool;
    fn peek(&self) -> I;
    fn advance(&mut self);

    fn consume(&mut self) -> I {
        let item = self.peek();
        self.advance();
        item
    }

    fn transform(item: I) -> T;

    fn satisfies_predicate<F>(&mut self, predicate: F) -> bool
    where
        F: Fn(T) -> bool,
    {
        match predicate(Self::transform(self.peek())) {
            false => false,
            true => {
                self.advance();
                true
            }
        }
    }

    fn matches(&mut self, item: T) -> bool {
        self.satisfies_predicate(|peeked| item == peeked)
    }

    fn matches_one_of(&mut self, items: &[T]) -> bool {
        self.satisfies_predicate(|peeked| items.contains(&peeked))
    }

    fn not_matches(&mut self, item: T) -> bool {
        self.satisfies_predicate(|peeked| item != peeked)
    }

    fn matches_none_of(&mut self, items: &[T]) -> bool {
        self.satisfies_predicate(|peeked| !items.contains(&peeked))
    }
}

struct Lexer {
    source: String,
    start: usize,
    current: usize,
    tokens: Vec<Token>,
}

impl Analyzer<char, char> for Lexer {
    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn peek(&self) -> char {
        self.source.chars().nth(self.current).unwrap_or('\x00')
    }

    fn advance(&mut self) {
        self.current += 1
    }

    fn transform(item: char) -> char {
        item
    }
}

impl Lexer {
    fn create(source: String) -> Self {
        Self {
            source,
            start: 0,
            current: 0,
            tokens: vec![],
        }
    }

    fn synchronize_start(&mut self) {
        self.start = self.current
    }

    fn get_current_lexeme(&self) -> String {
        self.source[self.start..self.current].to_string()
    }

    fn build_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            lexeme: self.get_current_lexeme(),
            position: self.start,
        }
    }

    fn scan_string(&mut self) -> Result<TokenKind, Error> {
        // TODO: handle escape sequences etc
        while self.matches_none_of(&['"', '\n', '\x00']) {}

        match self.matches('"') {
            false => Err(Error {
                kind: ErrKind::UnterminatedStringLiteral,
                position: self.start,
            }),
            true => Ok(TokenKind::String),
        }
    }

    fn scan_character(&mut self) -> Result<TokenKind, Error> {
        // TODO: handle escape sequences etc
        while self.matches_none_of(&['\'', '\n', '\x00']) {}

        match self.matches('\'') {
            false => Err(Error {
                kind: ErrKind::InvalidCharacterLiteral {
                    reason: "non-terminated character literal",
                },
                position: self.start,
            }),
            true => {
                let lexeme = self.get_current_lexeme();
                let contents = lexeme.trim_start_matches('\'').trim_end_matches('\'');

                match contents.len() {
                    1 => Ok(TokenKind::Char),
                    _ => Err(Error {
                        kind: ErrKind::InvalidCharacterLiteral {
                            reason: "a character literal must have a length of 1",
                        },
                        position: self.start,
                    }),
                }
            }
        }
    }

    fn scan_integer(&mut self) -> Result<TokenKind, Error> {
        while self.satisfies_predicate(|char| char.is_ascii_digit()) {}

        Ok(TokenKind::Integer)
    }

    fn scan_keyword_or_integer(&mut self) -> TokenKind {
        while self.satisfies_predicate(|char| char.is_identifier()) {}

        TokenKind::get_keyword(self.get_current_lexeme()).unwrap_or(TokenKind::Identifier)
    }

    fn scan_token(&mut self) -> Result<Token, Error> {
        let mut token_kind: Option<TokenKind> = None;

        while token_kind.is_none() {
            match self.consume() {
                '\0' => token_kind = Some(TokenKind::Eof),
                ' ' | '\r' | '\t' | '\n' => {
                    self.synchronize_start();
                    continue;
                }
                ':' if self.matches('=') => token_kind = Some(TokenKind::ColonEqual),
                '!' if self.matches('=') => token_kind = Some(TokenKind::BangEqual),
                '>' if self.matches('=') => token_kind = Some(TokenKind::GreaterEqual),
                '<' if self.matches('=') => token_kind = Some(TokenKind::LessEqual),
                '-' if self.matches('>') => token_kind = Some(TokenKind::MinusGreater),
                '(' if self.matches(')') => token_kind = Some(TokenKind::Unit),
                '[' => token_kind = Some(TokenKind::BracketLeft),
                ']' => token_kind = Some(TokenKind::BracketRight),
                '(' => token_kind = Some(TokenKind::ParenLeft),
                ')' => token_kind = Some(TokenKind::ParenRight),
                '.' => token_kind = Some(TokenKind::Dot),
                ',' => token_kind = Some(TokenKind::Comma),
                ';' => token_kind = Some(TokenKind::Semicolon),
                ':' => token_kind = Some(TokenKind::Colon),
                '=' => token_kind = Some(TokenKind::Equal),
                '>' => token_kind = Some(TokenKind::Greater),
                '<' => token_kind = Some(TokenKind::Less),
                '+' => token_kind = Some(TokenKind::Plus),
                '-' => token_kind = Some(TokenKind::Minus),
                '*' => token_kind = Some(TokenKind::Asterisk),
                '/' => token_kind = Some(TokenKind::Slash),
                '%' => token_kind = Some(TokenKind::Percent),
                '|' => token_kind = Some(TokenKind::VerticalBar),
                '"' => token_kind = Some(self.scan_string()?),
                '\'' => token_kind = Some(self.scan_character()?),
                '0'..='9' => token_kind = Some(self.scan_integer()?),
                char if char.is_identifier() => token_kind = Some(self.scan_keyword_or_integer()),
                char => {
                    return Err(Error {
                        kind: ErrKind::UnrecognizedToken {
                            token: char,
                            potential_matches: vec![],
                        },
                        position: self.start,
                    })
                }
            }
        }

        Ok(self.build_token(token_kind.unwrap()))
    }

    fn run(&mut self) -> Result<(), Error> {
        while !self.is_at_end() {
            self.synchronize_start();

            let token = self.scan_token()?;
            self.tokens.push(token);
        }

        self.synchronize_start();
        self.tokens.push(self.build_token(TokenKind::Eof));

        Ok(())
    }

    fn tokens(self) -> Vec<Token> {
        self.tokens
    }
}

#[derive(Clone, Debug)]
struct OldParser {
    tokens: Vec<Token>,
    start_stack: Vec<usize>,
    current: usize,
    unrecoverable: bool,
}

impl Analyzer<Token, TokenKind> for OldParser {
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    fn peek(&self) -> Token {
        self.tokens[self.current].clone()
    }

    fn advance(&mut self) {
        self.current += 1
    }

    fn transform(item: Token) -> TokenKind {
        item.kind
    }
}

impl OldParser {
    fn create(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            start_stack: vec![],
            current: 0,
            unrecoverable: false,
        }
    }

    fn start_rule(&mut self) {
        self.start_stack.push(self.current);
    }

    fn finalize_node<T>(&mut self, node: T) -> T {
        self.pop_stack();

        node
    }

    fn abort_rule(&mut self, error: Error) -> Error {
        self.backtrack();

        error
    }

    fn backtrack(&mut self) {
        match self.start_stack.pop() {
            None => self.unrecoverable = true,
            Some(index) => self.current = index,
        }
    }

    fn pop_stack(&mut self) {
        let new_length = self.start_stack.len().saturating_sub(1);
        self.start_stack.truncate(new_length);
    }

    fn previous(&self) -> Token {
        self.tokens[self.current - 1].clone()
    }

    fn parse_prim_expr(&mut self) -> Result<Expr, Error> {
        if self.matches(TokenKind::Integer) {
            return Ok(Expr::IntegerExpr {
                token: self.previous(),
            });
        }

        if self.matches(TokenKind::Char) {
            return Ok(Expr::CharExpr {
                token: self.previous(),
            });
        }

        if self.matches(TokenKind::String) {
            return Ok(Expr::StringExpr {
                token: self.previous(),
            });
        }

        if self.matches(TokenKind::Identifier) {
            return Ok(Expr::IdentExpr {
                identifier: self.previous().lexeme,
            });
        }

        Err(Error {
            kind: ErrKind::ExpectedExpression,
            position: self.current,
        })
    }

    fn parse_grouping_expr(&mut self) -> Result<Expr, Error> {
        self.start_rule();

        if self.matches(TokenKind::ParenLeft) {
            let left_paren = self.previous();
            let grouped = self.parse_expr().map_err(|error| self.abort_rule(error))?;

            match self.matches(TokenKind::ParenRight) {
                true => Ok(Expr::GroupingExpr {
                    grouped: Box::new(grouped),
                })
                .map(|node| self.finalize_node(node)),
                false => {
                    self.backtrack();
                    Err(Error {
                        kind: ErrKind::UnmatchedParenthesis { left_paren },
                        position: self.current,
                    })
                }
            }
        } else {
            self.parse_prim_expr().map(|node| self.finalize_node(node))
        }
    }

    fn parse_binary_product_expr(&mut self) -> Result<Expr, Error> {
        self.start_rule();

        let mut expr = self
            .parse_grouping_expr()
            .map_err(|error| self.abort_rule(error))?;

        while self.matches_one_of(&[TokenKind::Asterisk, TokenKind::Slash, TokenKind::Percent]) {
            let operator = self.previous().kind;
            let right = self
                .parse_grouping_expr()
                .map_err(|error| self.abort_rule(error))?;

            expr = Expr::BinaryExpr {
                operator,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr).map(|node| self.finalize_node(node))
    }

    fn parse_binary_sum_expr(&mut self) -> Result<Expr, Error> {
        self.start_rule();

        let mut expr = self
            .parse_binary_product_expr()
            .map_err(|error| self.abort_rule(error))?;

        while self.matches_one_of(&[TokenKind::Plus, TokenKind::Minus]) {
            let operator = self.previous().kind;
            let right = self
                .parse_binary_product_expr()
                .map_err(|error| self.abort_rule(error))?;

            expr = Expr::BinaryExpr {
                operator,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr).map(|node| self.finalize_node(node))
    }

    fn parse_expr(&mut self) -> Result<Expr, Error> {
        self.start_rule();

        self.parse_binary_sum_expr()
            .map(|node| self.finalize_node(node))
            .map_err(|error| self.abort_rule(error))
    }

    fn parse_expr_stmt(&mut self) -> Result<Stmt, Error> {
        self.start_rule();

        Ok(Stmt::ExprStmt {
            expr: self.parse_expr().map_err(|error| self.abort_rule(error))?,
        })
        .map(|node| self.finalize_node(node))
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Error> {
        self.parse_expr_stmt()
    }

    fn parse_ast(&mut self) -> Result<Ast, Error> {
        let mut body: Vec<Stmt> = vec![];

        while !self.is_at_end() {
            body.push(self.parse_stmt()?);
        }

        Ok(Ast { body })
    }

    fn run(&mut self) -> Result<Ast, Error> {
        self.parse_ast()
    }
}

trait Parser: StmtParser + ExprParser {
    fn parse_expr(&mut self) -> Result<Expr, Error> {
        self.parse_compound_expr()
    }

    fn parse_stmt(&mut self) -> Option<Result<Stmt, Error>>;
}

trait StmtParser {
    fn parse_expr_stmt(&mut self) -> Result<Stmt, Error>;
    fn parse_let_stmt(&mut self) -> Result<Stmt, Error>;
    fn parse_type_stmt(&mut self) -> Result<Stmt, Error>;
}
trait ExprParser {
    fn parse_atom_expr(&mut self) -> Result<Expr, Error>;
    fn parse_prioritized_expr(&mut self) -> Result<Expr, Error>;
    fn parse_extraction_expr(&mut self) -> Result<Expr, Error>;
    fn parse_application_expr(&mut self) -> Result<Expr, Error>;
    fn parse_unary_strong_expr(&mut self) -> Result<Expr, Error>;
    fn parse_unary_weak_expr(&mut self) -> Result<Expr, Error>;
    fn parse_binary_expr(&mut self) -> Result<Expr, Error>;
    fn parse_labeling_expr(&mut self) -> Result<Expr, Error>;
    fn parse_control_flow_expr(&mut self) -> Result<Expr, Error>;
    fn parse_compound_expr(&mut self) -> Result<Expr, Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Precedence {
    // Identifiers, numbers, strings, etc
    Atom,
    // Grouping
    Prioritized,
    // Membership
    Extraction,
    // Function calls, etc
    Application,
    // We have 2 precedence orders for unary to handle the case
    // of having both prefix and postfix on the same expression
    UnaryStrong,
    UnaryWeak,
    // rank 0 is sum, rank 1 is product, and so on
    Binary { rank: usize },
    // Application labels tie very loosely
    Labeling,
    // Match and alike
    ControlFlow,
    // Chained expressions
    Compound,
}

trait StmtRuleParser {
    fn parse(&self, parent: &dyn StmtParser) -> Option<Result<Stmt, Error>>;
}

trait ExprRuleParser {
    fn precedence(&self) -> &Precedence;
    fn parse(&self, parent: &dyn ExprParser) -> Option<Result<Expr, Error>>;
}

struct GlobalParser {
    tokens: Vec<Token>,
    stmt_parsers: Vec<&'static dyn StmtRuleParser>,
    expr_parsers: Vec<&'static dyn ExprRuleParser>,
    start_stack: Vec<usize>,
    current: usize,
    unrecoverable: bool,
    allow_unimplemented: bool,
}

impl Parser for GlobalParser {
    fn parse_stmt(&mut self) -> Option<Result<Stmt, Error>> {
        for parser in self.stmt_parsers.iter() {
            match parser.parse(self) {
                None => continue,
                Some(result) => return Some(result),
            }
        }

        None
    }
}

impl StmtParser for GlobalParser {
    fn parse_expr_stmt(&mut self) -> Result<Stmt, Error> {
        todo!()
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, Error> {
        todo!()
    }

    fn parse_type_stmt(&mut self) -> Result<Stmt, Error> {
        todo!()
    }
}

impl ExprParser for GlobalParser {
    fn parse_atom_expr(&mut self) -> Result<Expr, Error> {
        // TODO: instead of generating the error here, use Option
        // to make it climb such that parse_expr() is the one producing
        // the "expected expression" error
        self.parse_expr_or(
            Precedence::Atom,
            Err(Error {
                kind: ErrKind::ExpectedExpression,
                position: self.current,
            }),
        )
    }

    fn parse_prioritized_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::Prioritized, |parser| parser.parse_atom_expr())
    }

    fn parse_extraction_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::Extraction, |parser| {
            parser.parse_prioritized_expr()
        })
    }

    fn parse_application_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::Application, |parser| {
            parser.parse_extraction_expr()
        })
    }

    fn parse_unary_strong_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::UnaryStrong, |parser| {
            parser.parse_application_expr()
        })
    }

    fn parse_unary_weak_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::UnaryWeak, |parser| {
            parser.parse_unary_strong_expr()
        })
    }

    fn parse_binary_expr(&mut self) -> Result<Expr, Error> {
        todo!()
    }

    fn parse_labeling_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::Labeling, |parser| parser.parse_binary_expr())
    }

    fn parse_control_flow_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::ControlFlow, |parser| {
            parser.parse_labeling_expr()
        })
    }

    fn parse_compound_expr(&mut self) -> Result<Expr, Error> {
        self.parse_expr_or_else(Precedence::Compound, |parser| {
            parser.parse_control_flow_expr()
        })
    }
}

impl GlobalParser {
    fn create(tokens: Vec<Token>, allow_unimplemented: bool) -> Self {
        Self {
            tokens,
            stmt_parsers: vec![],
            expr_parsers: vec![],
            start_stack: vec![],
            current: 0,
            unrecoverable: false,
            allow_unimplemented,
        }
    }

    fn add_stmt_parser(&mut self, parser: &'static dyn StmtRuleParser) {
        self.stmt_parsers.push(parser);
    }

    fn add_expr_parser(&mut self, parser: &'static dyn ExprRuleParser) {
        self.expr_parsers.push(parser);
    }

    fn get_expr_parsers_with_precedence(
        &self,
        precedence: Precedence,
    ) -> Vec<&&'static dyn ExprRuleParser> {
        self.expr_parsers
            .iter()
            .filter(|parser| parser.precedence() == &precedence)
            .collect()
    }

    fn pop_stack(&mut self) {
        let new_length = self.start_stack.len().saturating_sub(1);
        self.start_stack.truncate(new_length);
    }

    fn backtrack(&mut self) {
        match self.start_stack.pop() {
            None => self.unrecoverable = true,
            Some(index) => self.current = index,
        }
    }

    fn start_rule(&mut self) {
        self.start_stack.push(self.current);
    }

    fn end_rule(&mut self, result: Result<Expr, Error>) -> Result<Expr, Error> {
        match result {
            Ok(_) => self.pop_stack(),
            Err(_) => self.backtrack(),
        }

        result
    }

    fn parse_expr_or(
        &mut self,
        precedence: Precedence,
        fallback: Result<Expr, Error>,
    ) -> Result<Expr, Error> {
        self.parse_expr_or_else(precedence, |_| fallback.clone())
    }

    fn parse_expr_or_else<F>(&mut self, precedence: Precedence, next_rule: F) -> Result<Expr, Error>
    where
        F: Fn(&mut Self) -> Result<Expr, Error>,
    {
        let parsers = self.get_expr_parsers_with_precedence(precedence);

        if !self.allow_unimplemented && parsers.is_empty() {
            return Err(Error {
                kind: ErrKind::UnimplementedExprRule {
                    precedence: precedence,
                },
                position: self.current,
            });
        }

        for parser in parsers {
            match parser.parse(self) {
                None => continue,
                Some(result) => return result,
            }
        }

        next_rule(self)
    }
}

fn lex(source: String) -> Result<Vec<Token>, Error> {
    let mut lexer = Lexer::create(source);
    lexer.run()?;

    Ok(lexer.tokens())
}

fn parse(tokens: Vec<Token>) -> Result<Ast, Error> {
    let mut parser = OldParser::create(tokens);

    parser.run()
}

fn compile(source: String) -> Result<Ast, Error> {
    lex(source).and_then(parse)
}

fn print_error(error: Error) {
    eprintln!("\x1b[1;31mError:\x1b[22;39m {:#?}", error);
}

fn main() {
    let source = "3 * 5".to_string();

    match compile(source) {
        Ok(ast) => {
            println!("{:#?}", ast);
        }
        Err(error) => print_error(error),
    }
}
