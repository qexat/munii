# munii

>[!CAUTION]
> Work in progress. Very much NOT usable currently

**munii** is an interactive, statically-typed array? functional? programming language. Its goal is to conciliate formal verification with quick prototyping.

Its particularity is that the process of getting code to run is bidirectional: instead of a pipeline compilation → execution, the runtime is able to communicate with the compiler in real time. This is due to the fact that the programming environment is not inside a file, but within the interactive shell **beqona**.

> [!TIP]
> You can find more information about it [here](./spec/index.md).

## Dev notes

Originally started in Python, drifted to OCaml then Rust 😅

---

- munii means eye
- beqona means conversation
