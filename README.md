<div align="center">

# ExprZ

[![Workflow Status](https://img.shields.io/github/workflow/status/qdeduction/exprz/main?label=workflow&style=flat-square)](https://github.com/qdeduction/exprz/actions)
[![Latest Version](https://img.shields.io/crates/v/exprz.svg?style=flat-square)](https://crates.io/crates/exprz)

_An Expression Library_

</div>

## About

ExprZ is an expression representation and parsing library. ExprZ expressions are typed s-expressions formed from `Atom`s or `Group`s of expressions and represent enumerated types of the form

```rust
enum Expr<A> {
    Atom(A),
    Group(Vec<Self>),
}
```

ExprZ generalizes this `enum` by defining an `Expression` trait which encompasses the algebraic properties of the above `enum` but which affords the user the flexibility of a more efficient implementation.

ExprZ comes with default implementations of `Expression`s which use the Rust `std` library. To access only the traits and algorithms through a `no_std` library see the [`core`](core) directory for the [`exprz-core`](https://docs.rs/exprz-core) package.

## Getting Started

For more information on how to use ExprZ, see the [documentation](https://docs.rs/exprz). 

## License

This project is licensed under the [ISC License](https://opensource.org/licenses/ISC). See [LICENSE](LICENSE) for more information.

---
<div align="center">

[![Author](https://img.shields.io/badge/-bhgomes-blue?style=for-the-badge)](https://github.com/bhgomes)
[![License](https://img.shields.io/badge/-LICENSE-lightgray?style=for-the-badge)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/-GitHub-black?style=for-the-badge)](https://github.com/qdeduction/exprz.rs)

</div>

