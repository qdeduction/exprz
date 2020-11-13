// file: src/lib.rs
// authors: Brandon H. Gomes

//! an expression library

#![no_std]

use core::iter::FromIterator;

///
///
pub trait RefIterator<T> {
    ///
    ///
    type Iter: Iterator<Item = T>;

    ///
    ///
    fn iter(&self) -> Self::Iter;
}

///
///
pub trait IntoRefIterator<T> {
    ///
    ///
    type RefIter: RefIterator<T>;

    ///
    ///
    fn ref_iter(&self) -> Self::RefIter;
}

/// Expression Tree
pub trait Expression
where
    Self: Into<Expr<Self>>,
{
    /// Atomic element type
    type Atom;

    /// Group expression type
    type Group: IntoRefIterator<Self>;

    ///
    ///
    fn cases(&self) -> ExprRef<Self>;

    /// Build an `Expression` from an atomic element.
    fn from_atom(atom: Self::Atom) -> Self;

    /// Build an `Expression` from a grouped expression.
    fn from_group(group: Self::Group) -> Self;

    /// Convert from [canonical enumeration]
    ///
    /// [canonical enumeration]: enum.Expr.html
    fn from_expr(expr: Expr<Self>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::from_atom(atom),
            Expr::Group(group) => Self::from_group(group),
        }
    }

    ///
    ///
    #[inline]
    fn clone(&self) -> Self
    where
        Self::Atom: Clone,
        Self::Group: FromIterator<Self>,
    {
        Self::from_expr(self.cases().into())
    }

    ///
    ///
    fn eq<Rhs>(&self, other: &Rhs) -> bool
    where
        Rhs: Expression,
        Self::Atom: PartialEq<Rhs::Atom>,
    {
        match (self.cases(), other.cases()) {
            (ExprRef::Atom(lhs), ExprRef::Atom(rhs)) => lhs == rhs,
            (ExprRef::Group(lhs), ExprRef::Group(rhs)) => Self::eq_group::<Self, Rhs>(lhs, rhs),
            _ => false,
        }
    }

    ///
    ///
    fn eq_group<L, R>(
        lhs: <L::Group as IntoRefIterator<L>>::RefIter,
        rhs: <R::Group as IntoRefIterator<R>>::RefIter,
    ) -> bool
    where
        L: Expression,
        R: Expression,
        L::Atom: PartialEq<R::Atom>,
    {
        let mut lhs = lhs.iter();
        let mut rhs = rhs.iter();
        loop {
            // FIXME: use nightly `eq_by`
            let x = match lhs.next() {
                None => return rhs.next().is_none(),
                Some(val) => val,
            };
            let y = match rhs.next() {
                None => return false,
                Some(val) => val,
            };
            if !x.eq(&y) {
                return false;
            }
        }
    }

    ///
    ///
    fn is_subexpression<Rhs>(&self, other: &Rhs) -> bool
    where
        Rhs: Expression,
        Self::Atom: PartialEq<Rhs::Atom>,
    {
        match self.cases() {
            ExprRef::Atom(atom) => match other.cases() {
                ExprRef::Atom(other) => atom == other,
                ExprRef::Group(other) => other.iter().any(move |e| self.is_subexpression(&e)),
            },
            ExprRef::Group(group) => match other.cases() {
                ExprRef::Atom(_) => false,
                ExprRef::Group(other) => {
                    other.iter().any(move |e| self.is_subexpression(&e))
                        || Self::eq_group::<Self, Rhs>(group, other)
                }
            },
        }
    }

    ///
    ///
    fn substitute<'s, I>(self, iter: &I) -> Self
    where
        Self::Atom: 's + PartialEq,
        Self::Group: FromIterator<Self>,
        I: RefIterator<(&'s Self::Atom, Self)>,
    {
        match self.into() {
            Expr::Atom(atom) => {
                piecewise_map(&atom, iter.iter()).unwrap_or_else(move || Self::from_atom(atom))
            }
            Expr::Group(group) => Self::from_group(
                group
                    .ref_iter()
                    .iter()
                    .map(move |e| e.substitute(iter))
                    .collect(),
            ),
        }
    }

    ///
    ///
    fn substitute_ref<'s, I>(&self, iter: &I) -> Self
    where
        Self: 's,
        Self::Atom: PartialEq + Clone,
        Self::Group: FromIterator<Self>,
        I: RefIterator<(&'s Self::Atom, &'s Self)>,
    {
        match self.cases() {
            ExprRef::Atom(atom) => piecewise_map(atom, iter.iter())
                .map_or_else(move || Self::from_atom(atom.clone()), Expression::clone),
            ExprRef::Group(group) => {
                Self::from_group(group.iter().map(move |e| e.substitute_ref(iter)).collect())
            }
        }
    }
}

fn piecewise_map<T, A, B, I>(t: T, iter: I) -> Option<B>
where
    T: PartialEq<A>,
    I: IntoIterator<Item = (A, B)>,
{
    // TODO: replace `find_map` body with `(t == a).then_some(b)` from nightly
    iter.into_iter()
        .find_map(move |(a, b)| if t == a { Some(b) } else { None })
}

///
///
pub enum ExprRef<'e, E>
where
    E: Expression,
{
    ///
    ///
    Atom(&'e E::Atom),

    ///
    ///
    Group(<E::Group as IntoRefIterator<E>>::RefIter),
}

impl<'e, E> ExprRef<'e, E>
where
    E: Expression,
{
    ///
    ///
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, ExprRef::Atom(_))
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, ExprRef::Group(_))
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<&'e E::Atom> {
        match self {
            ExprRef::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<<E::Group as IntoRefIterator<E>>::RefIter> {
        match self {
            ExprRef::Group(group) => Some(group),
            _ => None,
        }
    }

    ///
    ///
    #[inline]
    pub fn unwrap_atom(self) -> &'e E::Atom {
        self.atom().unwrap()
    }

    ///
    ///
    #[inline]
    pub fn unwrap_group(self) -> <E::Group as IntoRefIterator<E>>::RefIter {
        self.group().unwrap()
    }
}

/// Canonical Concrete Expression Type
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Expr<E>
where
    E: Expression,
{
    /// Atomic element
    Atom(E::Atom),

    /// Grouped expression
    Group(E::Group),
}

impl<E> From<E> for Expr<E>
where
    E: Expression,
{
    #[inline]
    fn from(e: E) -> Self {
        e.into()
    }
}

impl<'e, E> From<ExprRef<'e, E>> for Expr<E>
where
    E::Atom: Clone,
    E::Group: FromIterator<E>,
    E: Expression,
{
    #[inline]
    fn from(expr_ref: ExprRef<'e, E>) -> Self {
        match expr_ref {
            ExprRef::Atom(atom) => Self::from_atom(atom.clone()),
            ExprRef::Group(group) => {
                Self::from_group(group.iter().map(move |e| e.clone()).collect())
            }
        }
    }
}

impl<E> Expression for Expr<E>
where
    E: Expression,
{
    type Atom = E::Atom;

    type Group = E::Group;

    fn cases(&self) -> ExprRef<Self> {
        match self {
            Self::Atom(atom) => ExprRef::Atom(atom),
            Self::Group(group) => ExprRef::Group(ExprRefIterContainer {
                iter: group.ref_iter(),
            }),
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

impl<E> Expr<E>
where
    E: Expression,
{
    ///
    ///
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Expr::Atom(_))
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, Expr::Group(_))
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<E::Atom> {
        match self {
            Expr::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    ///
    ///
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<E::Group> {
        match self {
            Expr::Group(group) => Some(group),
            _ => None,
        }
    }

    ///
    ///
    #[inline]
    pub fn unwrap_atom(self) -> E::Atom {
        self.atom().unwrap()
    }

    ///
    ///
    #[inline]
    pub fn unwrap_group(self) -> E::Group {
        self.group().unwrap()
    }
}

///
///
pub struct ExprRefIterContainer<E>
where
    E: Expression,
{
    iter: <E::Group as IntoRefIterator<E>>::RefIter,
}

///
///
pub struct ExprRefIter<E>
where
    E: Expression,
{
    iter: <<E::Group as IntoRefIterator<E>>::RefIter as RefIterator<E>>::Iter,
}

impl<E> IntoRefIterator<Expr<E>> for E::Group
where
    E: Expression,
{
    type RefIter = ExprRefIterContainer<E>;

    fn ref_iter(&self) -> Self::RefIter {
        ExprRefIterContainer {
            iter: self.ref_iter(),
        }
    }
}

impl<E> RefIterator<Expr<E>> for ExprRefIterContainer<E>
where
    E: Expression,
{
    type Iter = ExprRefIter<E>;

    fn iter(&self) -> Self::Iter {
        ExprRefIter {
            iter: self.iter.iter(),
        }
    }
}

impl<E> Iterator for ExprRefIter<E>
where
    E: Expression,
{
    type Item = Expr<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(E::into)
    }
}
