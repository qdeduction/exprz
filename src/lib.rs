// file: src/lib.rs
// authors: Brandon H. Gomes

//! An Expression Library

#![feature(generic_associated_types)]
#![allow(incomplete_features)]

pub use exprz_core::*;

/// Vector Expressions
pub mod vec {
    use {
        super::{
            iter::{IntoIteratorGen, IteratorGen},
            ExprRef, Expression,
        },
        core::{iter::FromIterator, slice, str::FromStr},
        std::vec,
    };

    /// Vector Expression Type over `String`s
    pub type ExprString = Expr<String>;

    /// Vector Expression Type
    #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum Expr<A> {
        /// Atomic expression
        Atom(A),

        /// Grouped expression
        Group(Vec<Self>),
    }

    impl<A> Expression for Expr<A> {
        type Atom = A;

        type Group = ExprGroup<A>;

        fn cases(&self) -> ExprRef<Self> {
            match self {
                Self::Atom(atom) => ExprRef::Atom(atom),
                Self::Group(group) => ExprRef::Group(group),
            }
        }

        fn from_atom(atom: <Self as Expression>::Atom) -> Self {
            Self::Atom(atom)
        }

        fn from_group(group: <Self as Expression>::Group) -> Self {
            Self::Group(group.group)
        }
    }

    impl<A> Default for Expr<A> {
        #[inline]
        fn default() -> Self {
            <Self as Expression>::default()
        }
    }

    impl<A> FromStr for Expr<A> {
        type Err = super::parse::Error;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse()
        }
    }

    /// Vector Expression Group Wrapper Type
    pub struct ExprGroup<A> {
        /// Inner group
        pub group: Vec<Expr<A>>,
    }

    impl<A> IntoIterator for ExprGroup<A> {
        type Item = Expr<A>;

        type IntoIter = vec::IntoIter<Expr<A>>;

        fn into_iter(self) -> Self::IntoIter {
            self.group.into_iter()
        }
    }

    impl<A> FromIterator<Expr<A>> for ExprGroup<A> {
        fn from_iter<I>(iter: I) -> Self
        where
            I: IntoIterator<Item = Expr<A>>,
        {
            ExprGroup {
                group: iter.into_iter().collect(),
            }
        }
    }

    impl<A> IteratorGen<Expr<A>> for &Vec<Expr<A>> {
        type Item<'t>
        where
            A: 't,
        = &'t Expr<A>;

        type Iter<'t>
        where
            A: 't,
        = slice::Iter<'t, Expr<A>>;

        fn iter(&self) -> Self::Iter<'_> {
            (self[..]).iter()
        }
    }

    impl<A> IntoIteratorGen<Expr<A>> for ExprGroup<A> {
        type IterGen<'t>
        where
            A: 't,
        = &'t Vec<Expr<A>>;

        fn gen(&self) -> Self::IterGen<'_> {
            &self.group
        }
    }
}
