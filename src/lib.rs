// file: src/lib.rs
// authors: Brandon H. Gomes

//! An Expression Library

#![feature(generic_associated_types)]
#![allow(incomplete_features)]

pub use exprz_core::*;

/// Vector Expressions
pub mod vec {
    /// Vector Expression Type over `String`s
    pub type StringExpr = Expr<String>;

    /// Vector Expression Type
    #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum Expr<A> {
        /// Atomic expression
        Atom(A),

        /// Grouped expression
        Group(Vec<Self>),
    }
}

impl<A> Expression for vec::Expr<A> {
    type Atom = A;

    type Group = Vec<Self>;

    #[inline]
    fn cases(&self) -> ExprRef<Self> {
        match self {
            Self::Atom(atom) => ExprRef::Atom(atom),
            Self::Group(group) => ExprRef::Group(group),
        }
    }

    #[inline]
    fn from_atom(atom: <Self as Expression>::Atom) -> Self {
        Self::Atom(atom)
    }

    #[inline]
    fn from_group(group: <Self as Expression>::Group) -> Self {
        Self::Group(group)
    }
}

impl<A> From<vec::Expr<A>> for Expr<vec::Expr<A>> {
    #[inline]
    fn from(expr: vec::Expr<A>) -> Self {
        match expr {
            vec::Expr::Atom(atom) => Self::Atom(atom),
            vec::Expr::Group(group) => Self::Group(group),
        }
    }
}

impl<A> Default for vec::Expr<A> {
    #[inline]
    fn default() -> Self {
        <Self as Expression>::default()
    }
}

impl<A> iter::IteratorGen<vec::Expr<A>> for &Vec<vec::Expr<A>> {
    type Item<'t>
    where
        A: 't,
    = &'t vec::Expr<A>;

    type Iter<'t>
    where
        A: 't,
    = core::slice::Iter<'t, vec::Expr<A>>;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        (self[..]).iter()
    }
}

impl<A> iter::IntoIteratorGen<vec::Expr<A>> for Vec<vec::Expr<A>> {
    type IterGen<'t>
    where
        A: 't,
    = &'t Vec<vec::Expr<A>>;

    #[inline]
    fn gen(&self) -> Self::IterGen<'_> {
        &self
    }
}
