module Char = struct
  include Char

  let is_alpha : t -> bool = function
    | 'A' .. 'Z' | 'a' .. 'z' | '_' -> true
    | _ -> false
  ;;

  let is_digit : t -> bool = function
    | '0' .. '9' -> true
    | _ -> false
  ;;

  let is_alphanumeric (char : t) : bool = is_alpha char || is_digit char
end

module Iter = struct
  type ('a, 'b) t =
    | Base of 'a
    | Continue of 'b

  let return (value : 'a) : ('a, 'b) t = Base value
  let continue (value : 'b) : ('a, 'b) t = Continue value

  let rec loop (f : 'a -> ('b, 'a) t) (a : 'a) : 'b =
    match f a with
    | Base b -> b
    | Continue a' -> loop f a'
  ;;
end

module List = struct
  include List

  let contains : 'a t -> 'a -> bool = Fun.flip mem
end

module Option = struct
  include Option

  let ( or ) (option : 'a t) (default : 'a) : 'a = value ~default option
end

let ( or ) = Option.( or )

module TokenKind = struct
  type t =
    | IDENTIFIER
    | INTEGER
    | STRING
    | UNIT
    | DYNAMIC
    | FUN
    | LET
    | BRACKET_LEFT
    | BRACKET_RIGHT
    | PAREN_LEFT
    | PAREN_RIGHT
    | DOT
    | COMMA
    | SEMICOLON
    | EQUAL
    | COLON_EQUAL
    | COLON
    | BANG_EQUAL
    | GREATER
    | GREATER_EQUAL
    | LESS
    | LESS_EQUAL
    | PLUS
    | MINUS
    | ASTERISK
    | SLASH
    | PERCENT
    | VERTICAL_BAR
    | MINUS_GREATER
    | EOF

  let get_keyword : string -> t option = function
    | "dynamic" -> Some DYNAMIC
    | "fun" -> Some FUN
    | "let" -> Some LET
    | _ -> None
  ;;

  let render : t -> string = function
    | ASTERISK -> "ASTERISK"
    | BANG_EQUAL -> "BANG_EQUAL"
    | BRACKET_LEFT -> "BRACKET_LEFT"
    | BRACKET_RIGHT -> "BRACKET_RIGHT"
    | COLON -> "COLON"
    | COLON_EQUAL -> "COLON_EQUAL"
    | COMMA -> "COMMA"
    | DOT -> "DOT"
    | DYNAMIC -> "DYNAMIC"
    | EOF -> "EOF"
    | EQUAL -> "EQUAL"
    | FUN -> "FUN"
    | GREATER -> "GREATER"
    | GREATER_EQUAL -> "GREATER_EQUAL"
    | IDENTIFIER -> "IDENTIFIER"
    | INTEGER -> "INTEGER"
    | LESS -> "LESS"
    | LESS_EQUAL -> "LESS_EQUAL"
    | LET -> "LET"
    | MINUS -> "MINUS"
    | MINUS_GREATER -> "MINUS_GREATER"
    | PAREN_LEFT -> "PAREN_LEFT"
    | PAREN_RIGHT -> "PAREN_RIGHT"
    | PERCENT -> "PERCENT"
    | PLUS -> "PLUS"
    | SEMICOLON -> "SEMICOLON"
    | SLASH -> "SLASH"
    | STRING -> "STRING"
    | UNIT -> "UNIT"
    | VERTICAL_BAR -> "VERTICAL_BAR"
  ;;
end

module Token = struct
  type t =
    { kind : TokenKind.t
    ; lexeme : string
    ; position : int
    }

  let render : t -> string = function
    | { kind; lexeme; position } ->
      Printf.sprintf "%s '%s' %d" (TokenKind.render kind) (String.escaped lexeme) position
  ;;
end

module Err = struct
  type kind = ..

  type t =
    { kind : kind
    ; context : string
    ; position : int
    }
end

module Lexer = struct
  type Err.kind +=
    | InvalidCharacterLiteral of
        { lexeme : string
        ; reason : string
        }
    | UnrecognizedToken of
        { lexeme : string
        ; potential_matches : string list
        }
    | UnterminatedStringLiteral

  type t =
    { source : string
    ; start : int ref
    ; current : int ref
    ; tokens : Token.t list ref
    }

  let create (source : string) : t =
    { source; start = ref 0; current = ref 0; tokens = ref [] }
  ;;

  let is_at_end : t -> bool = function
    | { current; source; _ } -> !current >= String.length source
  ;;

  let synchronize_start : t -> unit = function
    | { start; current; _ } -> start := !current
  ;;

  let get_current_lexeme : t -> string = function
    | { source; start; current } -> String.sub source !start (!current - !start)
  ;;

  let build_token (lexer : t) (kind : TokenKind.t) : Token.t =
    { kind; lexeme = get_current_lexeme lexer; position = !(lexer.start) }
  ;;

  let peek (lexer : t) : string =
    match is_at_end lexer with
    | true -> "\x00"
    | false -> String.sub lexer.source !(lexer.current) 1
  ;;

  let advance : t -> unit = function
    | { current; _ } -> current := !current + 1
  ;;

  let consume (lexer : t) : string =
    let char = peek lexer in
    advance lexer;
    char
  ;;

  let satisfies_predicate (lexer : t) (predicate : string -> bool) : bool =
    match predicate (peek lexer) with
    | false -> false
    | true ->
      advance lexer;
      true
  ;;

  let matches (lexer : t) (char : string) : bool = satisfies_predicate lexer (( = ) char)

  let matches_one_of (lexer : t) (chars : string list) : bool =
    satisfies_predicate lexer (List.contains chars)
  ;;

  let not_matches (lexer : t) (char : string) : bool =
    satisfies_predicate lexer (( <> ) char)
  ;;

  let matches_none_of (lexer : t) (chars : string list) : bool =
    not (matches_one_of lexer chars)
  ;;

  let scan_string (lexer : t) : (TokenKind.t, Err.t) result =
    while matches_none_of lexer [ "\" "; "\n"; "\x00" ] do
      ()
    done;
    match matches lexer "\"" with
    | false ->
      Error
        { kind = UnterminatedStringLiteral
        ; context = lexer.source
        ; position = !(lexer.start)
        }
    | true -> Ok TokenKind.STRING
  ;;

  let scan_integer (lexer : t) : (TokenKind.t, Err.t) result =
    while satisfies_predicate lexer (String.for_all Char.is_digit) do
      ()
    done;
    Ok TokenKind.INTEGER
  ;;

  let scan_keyword_or_identifier (lexer : t) : (TokenKind.t, Err.t) result =
    while satisfies_predicate lexer (String.for_all Char.is_alphanumeric) do
      ()
    done;
    Ok (get_current_lexeme lexer |> TokenKind.get_keyword or TokenKind.IDENTIFIER)
  ;;

  let scan_token (lexer : t) : (Token.t, Err.t) result =
    let open Iter in
    loop
      (fun () : ((TokenKind.t, Err.t) result, unit) Iter.t ->
        match consume lexer with
        | "\x00" -> return (Ok TokenKind.EOF)
        | " " | "\r" | "\t" | "\n" ->
          synchronize_start lexer;
          continue ()
        | ":" when matches lexer "=" -> return (Ok TokenKind.COLON_EQUAL)
        | "!" when matches lexer "=" -> return (Ok TokenKind.BANG_EQUAL)
        | ">" when matches lexer "=" -> return (Ok TokenKind.GREATER_EQUAL)
        | "<" when matches lexer "=" -> return (Ok TokenKind.LESS_EQUAL)
        | "-" when matches lexer ">" -> return (Ok TokenKind.MINUS_GREATER)
        | "(" when matches lexer ")" -> return (Ok TokenKind.UNIT)
        | "[" -> return (Ok TokenKind.BRACKET_LEFT)
        | "]" -> return (Ok TokenKind.BRACKET_RIGHT)
        | "(" -> return (Ok TokenKind.PAREN_LEFT)
        | ")" -> return (Ok TokenKind.PAREN_RIGHT)
        | "." -> return (Ok TokenKind.DOT)
        | "," -> return (Ok TokenKind.COMMA)
        | ";" -> return (Ok TokenKind.SEMICOLON)
        | ":" -> return (Ok TokenKind.COLON)
        | "=" -> return (Ok TokenKind.EQUAL)
        | ">" -> return (Ok TokenKind.GREATER)
        | "<" -> return (Ok TokenKind.LESS)
        | "+" -> return (Ok TokenKind.PLUS)
        | "-" -> return (Ok TokenKind.MINUS)
        | "*" -> return (Ok TokenKind.ASTERISK)
        | "/" -> return (Ok TokenKind.SLASH)
        | "%" -> return (Ok TokenKind.PERCENT)
        | "|" -> return (Ok TokenKind.VERTICAL_BAR)
        | "\"" -> return (scan_string lexer)
        | char when (String.for_all Char.is_digit) char -> return (scan_integer lexer)
        | char when (String.for_all Char.is_alpha) char ->
          return (scan_keyword_or_identifier lexer)
        | char ->
          return
            (Error
               { Err.kind = UnrecognizedToken { lexeme = char; potential_matches = [] }
               ; context = lexer.source
               ; position = !(lexer.start)
               }))
      ()
    |> Result.map (build_token lexer)
  ;;

  let run (lexer : t) : (Token.t list, Err.t) result =
    let aux () =
      Iter.loop
        (fun () : ((unit, Err.t) result, unit) Iter.t ->
          match is_at_end lexer with
          | true -> Iter.return (Ok ())
          | false ->
            (match scan_token lexer with
             | Ok token ->
               lexer.tokens := token :: !(lexer.tokens);
               Iter.continue ()
             | Error error -> Iter.return (Error error)))
        ()
    in
    match aux () with
    | Error error -> Error error
    | Ok () ->
      synchronize_start lexer;
      lexer.tokens := build_token lexer TokenKind.EOF :: !(lexer.tokens);
      Ok (List.rev !(lexer.tokens))
  ;;
end

let lex (source : string) : (Token.t list, Err.t) result =
  Lexer.create source |> Lexer.run
;;

let print_error (error : Err.t) : unit = ()
let source = "let x = 3"

let tokens =
  match lex source with
  | Error error ->
    print_error error;
    exit 1
  | Ok tokens -> tokens
;;

List.iter (fun token -> Printf.printf "%s\n" (Token.render token)) tokens
