# Bidirectionality

The workflow of compiled programming languages is commonly a pipeline: first, the programmer uses a compiler to transform their code into some executable ; it is then passed to a runtime, whether the user runs their program manually or a tool does it for them. Once the compiler's job is done, it never sees that code again (unless, of course, the user recompiles without doing any change).

**munii** stands out because its runtime collects the environment values, and feeds them back to the compiler so the latter can refine its analysis of the past and future code. This is possible because in the language, values can be types.

## The munii pipeline

It looks like this:

```ocaml
Static Type Checking (1) -> Execution (2) -> Type Inference Refinement (3)
```

> Parts that are not useful in this article such as parsing have been stripped out.

### Example

First, we need some setup code to avoid using literals which don't really convey the point since their exact value is statically known.

```ocaml
(* in message.txt *)
munii is pretty cool!
```

```ocaml
(* in the shell *)
munii > let file = try File.open "message.txt" (mode := FileMode.READ)
______________________________________________
message : File
```

Then, let's read the contents of the file and store it in a variable.

```ocaml
munii > let contents = File.read_all file
_________________________________________
contents : "munii is pretty cool!\n"
```

Here, the trick is hidden: the static analysis of `contents` actually reveals the type `String` (1). Then, the execution happens without trouble, and the runtime reports the value of `contents` back to the compiler (2). Finally, the latter is allowed to refine its analysis by inferring the value `"munii is pretty cool!\n"` as the type of `contents` (3).

Now, we are going to define a function that strips the trailing newline.

>[!NOTE]
> This article is mostly about compiler-runtime bidirectionality, but we also use this opportunity to showcase some advanced type features such as dependent typing.

```ocaml
munii > let strip_trailing_newline string =
      |   match string
      |     case rest ++ "\n" -> rest
      |   end
________________________________________________________________
strip_trailing_newline : (string : rest ++ "\n") -> rest
```

Let's try it out:

```ocaml
munii > let without_newline = strip_trailing_newline contents
______________________________________________________________________
without_newline : "munii is pretty cool!"
```

Here, the type inference engine was able to figure out the exact type of `without_newline` without even needing the runtime's feedback!

However, the error case is probably more interesting:

```ocaml
munii > let really_no_newline = strip_trailing_newline without_newline
______________________________________________________________________
Error: the type of argument `without_newline` does not match expectations:

    1  | let really_no_newline = strip_trailing_newline without_newline
                                 ---------------------- ^^^^^^^^^^^^^^^
  -> `strip_trailing_newline` first argument was expected to be a `rest ++ "\n"`
  -> `without_newline` fails to match such type

Failed to define `really_no_newline`.
```

To be clear, this error is not a runtime one: it was thrown during static type analysis.

> [!TIP]
> Feedback is essentially the runtime providing proofs to the compiler!
