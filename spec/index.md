# Specification for the munii programming language

This documentation aims to provide a specification for the munii programming language as well as its tooling.

## Munii

*munii* is an [**interactive**](#interactive), [**statically-typed**](#statically-typed), [**functional**](#functional) programming language.

```sh
munii> return "Hello, munii!"
Hello, munii!
```

### Interactive

**munii** consists of a kernel with which the user interacts via a shell called **beqona**. From their perspective, there is no boundary between compilation and runtime: upon entering an expression, feedback is presented to them, in the form of either a value, the report of a successful definition, or an error. This *REPL* (Read-Eval-Print Loop) entertains the idea that there is a bidirectional communication between the compiler and the runtime.

> [!TIP]
> More information about bidirectionality can be found [here](./bidirectionality.md).

### Statically typed

Despite this seemingly absence of barrier between compilation and runtime, which makes **munii** look akin to languages like Python, the language is statically typed.

(TODO: brief explanation of why munii is statically typed)

### Functional

(TODO: brief explanation of why munii is functional)
