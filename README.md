<div align="center">

<a href="https://github.com/qdeduction/exprz">
    <img src="https://raw.githubusercontent.com/qdeduction/exprz/master/assets/logo.svg" width="200em">
</a>

# ExprZ

[![Workflow Status](https://flat.badgen.net/github/checks/qdeduction/exprz?label=workflow)](https://github.com/qdeduction/exprz/actions)
[![Project Crate](https://flat.badgen.net/crates/v/exprz)](https://crates.io/crates/exprz)
[![Documentation](https://flat.badgen.net/badge/docs/latest/blue)](https://docs.rs/exprz)

_An Expression Library_

</div>

## About

_ExprZ_ is an expression representation and parsing library. _ExprZ_ expressions are typed s-expressions formed from atoms or groups of expressions and represent enumerated types of the form

```rust
enum Expr<A> {
    Atom(A),
    Group(Vec<Self>),
}
```

_ExprZ_ generalizes this `enum` by defining an `Expression` trait which encompasses the algebraic properties of the above `enum` but which affords the user the flexibility of a more efficient implementation.

_ExprZ_ comes with default implementations of `Expression` which use the Rust `alloc` library. To access these implementations use the `alloc` feature on the crate.

### Rust Nightly

The most recent version of _ExprZ_ uses the Rust nightly compiler toolchain which is necessary for defining the main `Expression` trait. In the future we hope to remove this requirement as the `Expression` API becomes more refined over time or once the standard compiler toolchain catches up.

## Getting Started

For more information on how to use _ExprZ_, see the [documentation](https://docs.rs/exprz). 

## License

This project is licensed under the [ISC License](https://opensource.org/licenses/ISC). See [LICENSE](LICENSE) for more information.

---
<div align="center">

[![Author](https://img.shields.io/badge/-AUTHOR-blue?style=for-the-badge)](https://github.com/bhgomes)
[![License](https://img.shields.io/badge/-LICENSE-lightgray?style=for-the-badge)](LICENSE)
[![Source](https://img.shields.io/badge/-SOURCE-black?style=for-the-badge)](https://github.com/qdeduction/exprz)

</div>

