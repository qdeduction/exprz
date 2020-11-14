// file: src/lib.rs
// authors: Brandon H. Gomes

//! An Expression Library

pub use exprz_core::*;

///
///
pub mod vec {
    use super::*;

    ///
    ///
    #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum Expr<A> {
        ///
        ///
        Atom(A),

        ///
        ///
        Group(Vec<Self>),
    }

    ///
    ///
    pub type ExprString = Expr<String>;

    impl<A> Expression for Expr<A>
    where
        A: 'static,
    {
        type Atom = A;
        type Group = ExprGroup<A>;

        fn cases(&self) -> ExprRef<Self> {
            match self {
                Self::Atom(atom) => ExprRef::Atom(atom),
                Self::Group(group) => todo!(),
            }
        }

        fn from_atom(atom: <Self as Expression>::Atom) -> Self {
            Self::Atom(atom)
        }

        fn from_group(group: <Self as Expression>::Group) -> Self {
            Self::Group(group.group)
        }
    }

    pub struct ExprGroup<A> {
        group: Vec<Expr<A>>,
    }

    pub struct ExprGroupIterGen<'a, A> {
        group: &'a ExprGroup<A>,
    }

    pub struct ExprGroupIter<A> {
        group: Vec<Expr<A>>,
    }

    impl<A> Iterator for ExprGroupIter<A> {
        type Item = Expr<A>;

        fn next(&mut self) -> Option<Self::Item> {
            todo!()
        }
    }

    impl<'a, A> iter::IteratorGen<Expr<A>> for ExprGroupIterGen<'a, A> {
        type Iter = ExprGroupIter<A>;

        fn iter(&self) -> Self::Iter {
            todo!()
        }
    }

    impl<A> iter::IntoIteratorGen<Expr<A>> for ExprGroup<A>
    where
        A: 'static,
    {
        type IterGen = ExprGroupIterGen<'static, A>;

        fn gen(&self) -> Self::IterGen {
            todo!()
        }
    }
}
