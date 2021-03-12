//! An Expression Library

#![cfg_attr(docsrs, feature(doc_cfg), deny(broken_intra_doc_links))]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]
#![forbid(unsafe_code)]
#![no_std]

use {
    crate::iter::*,
    core::{borrow::Borrow, iter::FromIterator},
};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "parse")]
use core::str::FromStr;

/// Package Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// FIXME: Remove `IntoIteratorGen/IteratorGen` by using something like
//
// ```
// for<'s> &'s Self::Group: IntoIterator<Item = &'s Self>,
// ```
//
// in the definition of `Expression`.

/// Expression Trait
pub trait Expression
where
    Self: Into<Expr<Self>>,
{
    /// Atomic Element Type
    type Atom;

    /// Group Expression Type
    type Group: IntoIteratorGen<Self>;

    /// Gets a reference to the underlying `Expression` type.
    fn cases(&self) -> ExprRef<Self>;

    /// Builds an `Expression` from an atomic element.
    fn from_atom(atom: Self::Atom) -> Self;

    /// Builds an `Expression` from a grouped expression.
    fn from_group(group: Self::Group) -> Self;

    /// Converts from the [canonical enumeration].
    ///
    /// [canonical enumeration]: enum.Expr.html
    #[must_use]
    #[inline]
    fn from_expr(expr: Expr<Self>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::from_atom(atom),
            Expr::Group(group) => Self::from_group(group),
        }
    }

    /// Parses a string into an `Expression`.
    #[cfg(feature = "parse")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
    #[must_use]
    #[inline]
    fn from_str(s: &str) -> parse::Result<Self>
    where
        Self::Atom: FromIterator<char>,
        Self::Group: FromIterator<Self>,
    {
        parse::from_str(s)
    }

    /// Checks if the `Expression` is atomic.
    #[must_use]
    #[inline]
    fn is_atom(&self) -> bool {
        self.cases().is_atom()
    }

    /// Checks if the `Expression` is a grouped expression.
    #[must_use]
    #[inline]
    fn is_group(&self) -> bool {
        self.cases().is_group()
    }

    /// Converts from an `Expression` to an `Option<E::Atom>`.
    #[must_use]
    #[inline]
    fn atom(self) -> Option<Self::Atom> {
        self.into().atom()
    }

    /// Converts from an `Expression` to an `Option<E::Group>`.
    #[must_use]
    #[inline]
    fn group(self) -> Option<Self::Group> {
        self.into().group()
    }

    /// Returns the contained `Atom` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a `Group`.
    #[inline]
    #[track_caller]
    fn unwrap_atom(self) -> Self::Atom {
        self.into().unwrap_atom()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an `Atom`.
    #[inline]
    #[track_caller]
    fn unwrap_group(self) -> Self::Group {
        self.into().unwrap_group()
    }

    /// Builds an empty atomic expression.
    #[inline]
    fn empty_atom<T>() -> Self
    where
        Self::Atom: FromIterator<T>,
    {
        Self::from_atom(None.into_iter().collect())
    }

    /// Builds an empty grouped expression.
    #[inline]
    fn empty_group() -> Self
    where
        Self::Group: FromIterator<Self>,
    {
        Self::from_group(None.into_iter().collect())
    }

    /// Gets the default value of an `Expression`: the empty group.
    #[inline]
    fn default() -> Self
    where
        Self::Group: FromIterator<Self>,
    {
        Self::empty_group()
    }

    /// Clones an `Expression` that has `Clone`-able `Atom`s.
    #[inline]
    fn clone(&self) -> Self
    where
        Self::Atom: Clone,
        Self::Group: FromIterator<Self>,
    {
        Self::from_expr(self.cases().into())
    }

    /// Checks if two `Expression`s are equal using `PartialEq` on their `Atom`s.
    #[inline]
    fn eq<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
    {
        self.cases() == other.cases()
    }

    /// Checks if an `Expression` is a sub-tree of another `Expression` using `PartialEq` on their
    /// `Atom`s.
    #[inline]
    fn is_subexpression<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
    {
        self.cases().is_subexpression(&other.cases())
    }

    /// Checks if expression matches given `Pattern`.
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches<P>(&self, pattern: P) -> bool
    where
        P: pattern::Pattern<Self>,
    {
        pattern.matches(self)
    }

    /// Checks if `self` matches an equality pattern.
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches_equal<P>(&self, pattern: &P) -> bool
    where
        P: Expression,
        P::Atom: PartialEq<Self::Atom>,
    {
        self.matches(pattern::EqualExpressionPattern::new(pattern))
    }

    /// Checks if `self` matches a subexpression pattern.
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches_subexpression<P>(&self, pattern: &P) -> bool
    where
        P: Expression,
        P::Atom: PartialEq<Self::Atom>,
    {
        self.matches(pattern::SubExpressionPattern::new(pattern))
    }

    /// Checks if `self` matches a basic shape pattern.
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches_basic_shape<P>(&self, pattern: &P) -> bool
    where
        P: Expression<Atom = pattern::BasicShape>,
        P::Atom: PartialEq<Self::Atom>,
    {
        self.matches(pattern::BasicShapePattern::new(pattern))
    }

    /// Checks if `self` matches a wildcard expression.
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches_wildcard<W, P>(&self, is_wildcard: W, pattern: &P) -> bool
    where
        P: Expression,
        P::Atom: PartialEq<Self::Atom>,
        W: Fn(&P::Atom) -> bool,
    {
        self.matches(pattern::WildCardPattern::new(is_wildcard, pattern))
    }

    /// Extends a function on `Atom`s to a function on `Expression`s.
    #[inline]
    fn map<E, F>(self, f: F) -> E
    where
        Self::Group: IntoIterator<Item = Self>,
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(Self::Atom) -> E::Atom,
    {
        self.into().map(f)
    }

    /// Extends a function on `&Atom`s to a function on `&Expression`s.
    #[inline]
    fn map_ref<E, F>(&self, f: F) -> E
    where
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(&Self::Atom) -> E::Atom,
    {
        self.cases().map_ref(f)
    }

    /// Substitutes an `Expression` into each `Atom` of `self`.
    #[inline]
    fn substitute<F>(self, f: F) -> Self
    where
        Self::Group: FromIterator<Self> + IntoIterator<Item = Self>,
        F: FnMut(Self::Atom) -> Self,
    {
        self.into().substitute(f)
    }

    /// Substitutes an `Expression` into each `Atom` of `&self`.
    #[inline]
    fn substitute_ref<F>(&self, f: F) -> Self
    where
        Self::Group: FromIterator<Self>,
        F: FnMut(&Self::Atom) -> Self,
    {
        self.cases().substitute_ref(f)
    }
}

/// Group Type Wrapper Trait
pub trait HasGroupType<'e, E, T>
where
    E: Expression,
{
    /// Gets underlying group type.
    fn group_type(&self) -> &'e T;
}

/// Multi-Expression Trait
pub trait MultiExpression
where
    Self: Expression,
    for<'e> <Self::Group as IntoIteratorGen<Self>>::IterGen<'e>:
        HasGroupType<'e, Self, Self::GroupType>,
{
    /// Group Type
    type GroupType;

    /// Gets `Self::GroupType` if `Expression` is a group.
    fn group_type(&self) -> Option<&Self::GroupType> {
        self.cases().group().map(move |g| g.group_type())
    }
}

/// Internal Reference to an `Expression` Type
pub enum ExprRef<'e, E>
where
    E: 'e + Expression,
{
    /// Reference to an atomic expression
    Atom(&'e E::Atom),

    /// Grouped expression `IteratorGen`
    Group(<E::Group as IntoIteratorGen<E>>::IterGen<'e>),
}

impl<'e, E> ExprRef<'e, E>
where
    E: Expression,
{
    /// Checks if the `ExprRef` is atomic.
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Self::Atom(_))
    }

    /// Checks if the `ExprRef` is a grouped expression `IteratorGen`.
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, Self::Group(_))
    }

    /// Converts from an `ExprRef<E>` to an `Option<&E::Atom>`.
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<&'e E::Atom> {
        match self {
            Self::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an `ExprRef<E>` to an `Option<E::Group::IterGen>`.
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<<E::Group as IntoIteratorGen<E>>::IterGen<'e>> {
        match self {
            Self::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Returns the contained `Atom` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a `Group`.
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> &'e E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an `Atom`.
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> <E::Group as IntoIteratorGen<E>>::IterGen<'e> {
        self.group().unwrap()
    }

    /// Checks if an `Expression` is a sub-tree of another `Expression` using `PartialEq` on their
    /// `Atom`s.
    pub fn is_subexpression<'r, R>(&self, other: &ExprRef<'r, R>) -> bool
    where
        R: Expression,
        E::Atom: PartialEq<R::Atom>,
    {
        match self {
            Self::Atom(atom) => match other {
                ExprRef::Atom(other) => atom == other,
                ExprRef::Group(other) => other
                    .iter()
                    .any(move |e| self.is_subexpression(&e.borrow().cases())),
            },
            Self::Group(group) => match other {
                ExprRef::Atom(_) => false,
                ExprRef::Group(other) => {
                    other
                        .iter()
                        .any(move |e| self.is_subexpression(&e.borrow().cases()))
                        || eq_by(group.iter(), other.iter(), move |l, r| {
                            l.borrow().eq(r.borrow())
                        })
                }
            },
        }
    }

    /// Extends a function on `&Atom`s to a function on `&Expression`s.
    #[inline]
    pub fn map_ref<O, F>(&self, mut f: F) -> O
    where
        O: Expression,
        O::Group: FromIterator<O>,
        F: FnMut(&E::Atom) -> O::Atom,
    {
        self.map_ref_inner(&mut f)
    }

    #[inline]
    fn map_ref_inner<O, F>(&self, f: &mut F) -> O
    where
        O: Expression,
        O::Group: FromIterator<O>,
        F: FnMut(&E::Atom) -> O::Atom,
    {
        match self {
            Self::Atom(atom) => O::from_atom(f(atom)),
            Self::Group(group) => O::from_group(
                group
                    .iter()
                    .map(move |e| e.borrow().cases().map_ref_inner(f))
                    .collect(),
            ),
        }
    }

    /// Substitutes an `Expression` into each `Atom` of `&self`.
    #[inline]
    pub fn substitute_ref<F>(&self, mut f: F) -> E
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        self.substitute_ref_inner(&mut f)
    }

    #[inline]
    fn substitute_ref_inner<F>(&self, f: &mut F) -> E
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        match self {
            Self::Atom(atom) => f(atom),
            Self::Group(group) => E::from_group(
                group
                    .iter()
                    .map(move |e| e.borrow().cases().substitute_ref_inner(f))
                    .collect(),
            ),
        }
    }
}

impl<'l, 'r, L, R> PartialEq<ExprRef<'r, R>> for ExprRef<'l, L>
where
    L: Expression,
    R: Expression,
    L::Atom: PartialEq<R::Atom>,
{
    /// Checks if two `Expression`s are equal using `PartialEq` on their `Atom`s.
    fn eq(&self, other: &ExprRef<'r, R>) -> bool {
        match (self, other) {
            (Self::Atom(lhs), ExprRef::Atom(rhs)) => *lhs == *rhs,
            (Self::Group(lhs), ExprRef::Group(rhs)) => {
                eq_by(lhs.iter(), rhs.iter(), move |l, r| {
                    l.borrow().eq(r.borrow())
                })
            }
            _ => false,
        }
    }
}

/// Canonical Concrete `Expression` Type
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Expr<E>
where
    E: Expression,
{
    /// Atomic element
    Atom(E::Atom),

    /// Grouped expression
    Group(E::Group),
}

impl<'e, E> From<ExprRef<'e, E>> for Expr<E>
where
    E::Atom: Clone,
    E::Group: FromIterator<E>,
    E: Expression,
{
    #[must_use]
    #[inline]
    fn from(expr_ref: ExprRef<'e, E>) -> Self {
        match expr_ref {
            ExprRef::Atom(atom) => Self::Atom(atom.clone()),
            ExprRef::Group(group) => {
                Self::Group(group.iter().map(move |e| e.borrow().clone()).collect())
            }
        }
    }
}

#[cfg(feature = "parse")]
#[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
impl<E> FromStr for Expr<E>
where
    E: Expression,
    E::Atom: FromIterator<char>,
    E::Group: FromIterator<E>,
{
    type Err = parse::Error;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        E::from_str(s).map(E::into)
    }
}

impl<E> Expr<E>
where
    E: Expression,
{
    /// Checks if the `Expr` is atomic.
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Expr::Atom(_))
    }

    /// Checks if the `Expr` is a grouped expression.
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, Expr::Group(_))
    }

    /// Converts from an `Expr<E>` to an `Option<E::Atom>`.
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<E::Atom> {
        match self {
            Expr::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an `Expr<E>` to an `Option<E::Group>`.
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<E::Group> {
        match self {
            Expr::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Returns the contained `Atom` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a `Group`.
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an `Atom`.
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> E::Group {
        self.group().unwrap()
    }

    /// Extends a function on `Atom`s to a function on `Expression`s.
    #[inline]
    pub fn map<O, F>(self, mut f: F) -> O
    where
        E::Group: IntoIterator<Item = E>,
        O: Expression,
        O::Group: FromIterator<O>,
        F: FnMut(E::Atom) -> O::Atom,
    {
        self.map_inner(&mut f)
    }

    fn map_inner<O, F>(self, f: &mut F) -> O
    where
        E::Group: IntoIterator<Item = E>,
        O: Expression,
        O::Group: FromIterator<O>,
        F: FnMut(E::Atom) -> O::Atom,
    {
        match self {
            Expr::Atom(atom) => O::from_atom(f(atom)),
            Expr::Group(group) => O::from_group(
                group
                    .into_iter()
                    .map(move |e| e.into().map_inner(f))
                    .collect(),
            ),
        }
    }

    /// Substitutes an `Expression` into each `Atom` of `self`.
    #[inline]
    pub fn substitute<F>(self, mut f: F) -> E
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        self.substitute_inner(&mut f)
    }

    fn substitute_inner<F>(self, f: &mut F) -> E
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        match self {
            Expr::Atom(atom) => f(atom),
            Expr::Group(group) => E::from_group(
                group
                    .into_iter()
                    .map(move |e| e.into().substitute_inner(f))
                    .collect(),
            ),
        }
    }
}

impl<E> Default for Expr<E>
where
    E: Expression,
    E::Group: FromIterator<E>,
{
    /// Returns the empty group expression.
    fn default() -> Self {
        E::default().into()
    }
}

/* TODO: Is it possible to implement this?
impl<E> Expression for Expr<E>
where
    E: Expression,
{
    type Atom = E::Atom;

    type Group = E::Group;

    #[inline]
    fn cases(&self) -> ExprRef<Self> {
        match self {
            Self::Atom(atom) => ExprRef::Atom(atom),
            Self::Group(group) => ExprRef::Group(ExprIterContainer::new(group.gen())),
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

impl<E> IntoIteratorGen<Expr<E>> for E::Group
where
    E: Expression,
{
    type IterGen<'t>
    where
        E: 't,
    = ExprIterContainer<'t, E>;

    #[inline]
    fn gen(&self) -> Self::IterGen<'_> {
        ExprIterContainer::new(self.gen())
    }
}

impl<'e, E> IteratorGen<Expr<E>> for ExprIterContainer<'e, E>
where
    E: Expression,
{
    type Item<'t>
    where
        E: 't,
    = Expr<E>;

    type Iter<'t>
    where
        E: 't,
    = ExprIter<'t, E>;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        ExprIter::from_container(self)
    }
}

impl<'e, E> Iterator for ExprIter<'e, E>
where
    E: Expression,
{
    type Item = Expr<E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner_next()
    }
}

impl<E> Expr<Expr<E>>
where
    E: Expression,
{
    /// Monadic join for `Expr`.
    pub fn join(self) -> Expr<E> {
        match self {
            Self::Atom(atom) => Expr::Atom(atom),
            Self::Group(group) => Expr::from_group(group),
        }
    }
}
*/

/// Parsing Module
#[cfg(feature = "parse")]
#[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
pub mod parse {
    use {
        super::Expression,
        core::{
            iter::{from_fn, FromIterator, Peekable},
            result,
        },
    };

    /// `Expression` Parsing Error
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum Error {
        /// Multiple expressions at top level
        MultiExpr,

        /// No closing quote
        MissingQuote,

        /// Group was not closed
        OpenGroup,

        /// Group was not opened
        UnopenedGroup,

        /// Found an empty group that was not opened or closed
        BadEmptyGroup,

        /// Found leading whitespace
        LeadingWhitespace,

        /// Found trailing symbols
        TrailingSymbols,

        /// Group was opened when only an `Atom` was expected
        BadOpenGroup,

        /// Atom was started when only a `Group` was expected
        BadStartAtom,
    }

    /// `Expression` Parsing Result Type
    pub type Result<T> = result::Result<T, Error>;

    /// Meaningful Symbols for `Expression` Parsing
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum SymbolType {
        /// Whitespace
        Whitespace,

        /// Start of a group
        GroupOpen,

        /// End of a group
        GroupClose,

        /// Start/End of a quoted sub-string
        Quote,

        /// Other characters
        Other,
    }

    /// Parses an `Expression` from an `Iterator` over `collect`-able symbols.
    ///
    /// This function consumes the iterator expecting nothing before or after the parsed
    /// `Expression`.
    pub fn parse<I, F, E>(iter: I, classify: F) -> Result<E>
    where
        I: IntoIterator,
        F: Fn(&I::Item) -> SymbolType,
        E: Expression,
        E::Atom: FromIterator<I::Item>,
        E::Group: FromIterator<E>,
    {
        let mut iter = iter.into_iter().peekable();
        if let Some(true) = iter.peek().map(|p| classify(p) == SymbolType::Whitespace) {
            return Err(Error::LeadingWhitespace);
        }
        let expr = parse_continue(&mut iter, &classify);
        iter.next()
            .map(move |_| Err(Error::MultiExpr))
            .unwrap_or(expr)
    }

    /// Tries to parse an `Expression` from an `Iterator` over `collect`-able symbols.
    ///
    /// The iterator may still have elements remaining after parsing one `Group`.
    pub fn parse_continue<I, F, E>(iter: &mut Peekable<I>, classify: F) -> Result<E>
    where
        I: Iterator,
        F: Fn(&I::Item) -> SymbolType,
        E: Expression,
        E::Atom: FromIterator<I::Item>,
        E::Group: FromIterator<E>,
    {
        match iter.peek() {
            Some(peek) => match classify(&peek) {
                SymbolType::GroupClose => Err(Error::UnopenedGroup),
                SymbolType::GroupOpen => {
                    parse_group_continue::<_, _, E>(iter, &classify).map(E::from_group)
                }
                _ => parse_atom_continue(iter, &classify).map(E::from_atom),
            },
            _ => Ok(E::empty_atom()),
        }
    }

    /// Parses a `Group` from an `Iterator` over `collect`-able symbols.
    ///
    /// This function consumes the iterator expecting nothing before or after the parsed `Group`.
    pub fn parse_group<I, F, E>(iter: I, classify: F) -> Result<E::Group>
    where
        I: IntoIterator,
        F: Fn(&I::Item) -> SymbolType,
        E: Expression,
        E::Atom: FromIterator<I::Item>,
        E::Group: FromIterator<E>,
    {
        let mut iter = iter.into_iter().peekable();
        let group = parse_group_continue::<_, _, E>(&mut iter, &classify);
        iter.next()
            .map(move |_| Err(Error::TrailingSymbols))
            .unwrap_or(group)
    }

    /// Tries to parse a `Group` from an `Iterator` over `collect`-able symbols.
    ///
    /// The iterator may still have elements remaining after parsing one `Group`.
    pub fn parse_group_continue<I, F, E>(iter: &mut Peekable<I>, classify: &F) -> Result<E::Group>
    where
        I: Iterator,
        F: Fn(&I::Item) -> SymbolType,
        E: Expression,
        E::Atom: FromIterator<I::Item>,
        E::Group: FromIterator<E>,
    {
        match iter.peek() {
            Some(peek) => match classify(&peek) {
                SymbolType::Whitespace => return Err(Error::LeadingWhitespace),
                SymbolType::GroupClose => return Err(Error::UnopenedGroup),
                SymbolType::GroupOpen => {
                    let _ = iter.next();
                }
                _ => return Err(Error::BadStartAtom),
            },
            _ => return Err(Error::BadEmptyGroup),
        }
        from_fn(parse_group_continue_inner(iter, classify)).collect()
    }

    #[inline]
    fn parse_group_continue_inner<'f, I, F, E>(
        iter: &'f mut Peekable<I>,
        classify: &'f F,
    ) -> impl 'f + FnMut() -> Option<Result<E>>
    where
        I: Iterator,
        F: 'f + Fn(&I::Item) -> SymbolType,
        E: Expression,
        E::Atom: FromIterator<I::Item>,
        E::Group: FromIterator<E>,
    {
        move || loop {
            match iter.peek() {
                Some(peek) => match classify(&peek) {
                    SymbolType::Whitespace => {
                        let _ = iter.next();
                    }
                    SymbolType::GroupClose => {
                        let _ = iter.next();
                        return None;
                    }
                    SymbolType::GroupOpen => {
                        return Some(
                            parse_group_continue::<_, _, E>(iter, classify).map(E::from_group),
                        );
                    }
                    _ => return Some(parse_atom_continue(iter, classify).map(E::from_atom)),
                },
                _ => return Some(Err(Error::OpenGroup)),
            }
        }
    }

    /// Parses an `Atom` from an `Iterator` over `collect`-able symbols.
    ///
    /// This function consumes the iterator expecting nothing before or after the parsed `Atom`.
    pub fn parse_atom<I, F, A>(iter: I, classify: F) -> Result<A>
    where
        I: IntoIterator,
        F: Fn(&I::Item) -> SymbolType,
        A: FromIterator<I::Item>,
    {
        let mut iter = iter.into_iter().peekable();
        if let Some(peek) = iter.peek() {
            match classify(&peek) {
                SymbolType::Whitespace => return Err(Error::LeadingWhitespace),
                SymbolType::GroupClose => return Err(Error::UnopenedGroup),
                SymbolType::GroupOpen => return Err(Error::BadOpenGroup),
                _ => {}
            }
        }
        let atom = parse_atom_continue(&mut iter, &classify);
        iter.next()
            .map(move |_| Err(Error::TrailingSymbols))
            .unwrap_or(atom)
    }

    /// Tries to parse an `Atom` from an `Iterator` over `collect`-able symbols.
    ///
    /// The iterator may still have elements remaining after parsing one `Atom`.
    pub fn parse_atom_continue<I, F, A>(iter: &mut Peekable<I>, classify: &F) -> Result<A>
    where
        I: Iterator,
        F: Fn(&I::Item) -> SymbolType,
        A: FromIterator<I::Item>,
    {
        let mut inside_quote = false;
        let atom = from_fn(parse_atom_continue_inner(iter, classify, &mut inside_quote)).collect();
        if inside_quote {
            Err(Error::MissingQuote)
        } else {
            Ok(atom)
        }
    }

    #[inline]
    fn parse_atom_continue_inner<'f, I, F>(
        iter: &'f mut Peekable<I>,
        classify: &'f F,
        inside_quote: &'f mut bool,
    ) -> impl 'f + FnMut() -> Option<I::Item>
    where
        I: Iterator,
        F: 'f + Fn(&I::Item) -> SymbolType,
    {
        move || match iter.peek() {
            Some(peek) => {
                if *inside_quote {
                    if classify(&peek) == SymbolType::Quote {
                        *inside_quote = false;
                    }
                } else {
                    match classify(&peek) {
                        SymbolType::Quote => *inside_quote = true,
                        SymbolType::Other => {}
                        _ => return None,
                    }
                }
                iter.next()
            }
            _ => None,
        }
    }

    /// Returns the default classification for the `char` type.
    #[inline]
    pub fn default_char_classification(c: &char) -> SymbolType {
        match c {
            '(' => SymbolType::GroupOpen,
            ')' => SymbolType::GroupClose,
            '"' => SymbolType::Quote,
            c => {
                if c.is_whitespace() {
                    SymbolType::Whitespace
                } else {
                    SymbolType::Other
                }
            }
        }
    }

    /// Parses a string-like `Expression` from an iterator over characters.
    #[inline]
    pub fn from_chars<I, E>(iter: I) -> Result<E>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        parse(iter, default_char_classification)
    }

    /// Parses a string-like `Expression` from a string.
    #[inline]
    pub fn from_str<S, E>(s: S) -> Result<E>
    where
        S: AsRef<str>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        from_chars(s.as_ref().chars())
    }

    /// Parses a string-like expression `Group` from an iterator over characters.
    ///
    /// # Panics
    ///
    /// Panics if the parsing was a valid `Expression` but not a `Group`.
    #[inline]
    #[track_caller]
    pub fn from_chars_as_group<I, E>(iter: I) -> Result<E::Group>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        from_chars(Some('(').into_iter().chain(iter).chain(Some(')'))).map(E::unwrap_group)
    }

    /// Parses a string-like expression `Group` from a string.
    ///
    /// # Panics
    ///
    /// Panics if the parsing was a valid `Expression` but not a `Group`.
    #[inline]
    #[track_caller]
    pub fn from_str_as_group<S, E>(s: S) -> Result<E::Group>
    where
        S: AsRef<str>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        from_chars_as_group::<_, E>(s.as_ref().chars())
    }
}

/// Shape Module
#[cfg(feature = "shape")]
#[cfg_attr(docsrs, doc(cfg(feature = "shape")))]
pub mod shape {
    use {super::*, core::convert::TryFrom};

    // TODO: how to resolve this with the `Pattern` trait?

    /// Matcher Trait
    pub trait Matcher<E>
    where
        E: Expression,
    {
        /// Match Error Type
        type Error;

        /// Checks if the given atom matches the shape.
        #[must_use]
        fn matches_atom(atom: &E::Atom) -> Result<(), Self::Error>;

        /// Checks if the given group matches the shape.
        #[must_use]
        fn matches_group(
            group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>,
        ) -> Result<(), Self::Error>;

        /// Checks if the given expression matches the shape.
        #[must_use]
        #[inline]
        fn matches(expr: &E) -> Result<(), Self::Error> {
            match expr.cases() {
                ExprRef::Atom(atom) => Self::matches_atom(atom),
                ExprRef::Group(group) => Self::matches_group(group),
            }
        }
    }

    /// Shape Trait
    ///
    /// # Contract
    ///
    /// The following should hold for all `expr: E`:
    ///
    /// ```
    /// matches(&expr).err() == expr.try_into().err()
    /// ```
    ///
    /// but can be weakend to the following,
    ///
    /// ```
    /// matches(&expr).is_err() == expr.try_into().is_err()
    /// ```
    ///
    /// if it is impossible or inefficient to implement the stronger contract.
    pub trait Shape<E>
    where
        Self: Into<Expr<E>>
            + TryFrom<Expr<E>>
            + Matcher<E, Error = <Self as TryFrom<Expr<E>>>::Error>,
        E: Expression,
    {
    }
}

/// Pattern Module
#[cfg(feature = "pattern")]
#[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
pub mod pattern {
    use super::*;

    // TODO: how to resolve this with the `Shape` trait?

    /// Pattern Trait
    pub trait Pattern<E>
    where
        E: Expression,
    {
        /// Checks if the pattern matches an atom.
        fn matches_atom(&self, atom: &E::Atom) -> bool;

        /// Checks if the pattern matches a group.
        fn matches_group(&self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool;

        /// Checks if the pattern matches an expression.
        #[inline]
        fn matches(&self, expr: &E) -> bool {
            match expr.cases() {
                ExprRef::Atom(atom) => self.matches_atom(atom),
                ExprRef::Group(group) => self.matches_group(group),
            }
        }
    }

    /// Mutating Pattern Trait
    pub trait PatternMut<E>
    where
        E: Expression,
    {
        /// Checks if the pattern matches an atom.
        fn matches_atom(&mut self, atom: &E::Atom) -> bool;

        /// Checks if the pattern matches a group.
        fn matches_group(&mut self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool;

        /// Checks if the pattern matches an expression.
        #[inline]
        fn matches(&mut self, expr: &E) -> bool {
            match expr.cases() {
                ExprRef::Atom(atom) => self.matches_atom(atom),
                ExprRef::Group(group) => self.matches_group(group),
            }
        }
    }

    /// Equal Expression Pattern
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct EqualExpressionPattern<'p, P>(&'p P)
    where
        P: Expression;

    impl<'p, P> EqualExpressionPattern<'p, P>
    where
        P: Expression,
    {
        pub(crate) fn new(pattern: &'p P) -> Self {
            Self(pattern)
        }
    }

    impl<'p, P, E> Pattern<E> for EqualExpressionPattern<'p, P>
    where
        E: Expression,
        P: Expression,
        P::Atom: PartialEq<E::Atom>,
    {
        fn matches_atom(&self, atom: &E::Atom) -> bool {
            self.0.cases().atom().map_or(false, |a| a == atom)
        }

        fn matches_group(&self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool {
            self.0.cases().group().map_or(false, move |g| {
                eq_by(g.iter(), group.iter(), move |p, e| {
                    p.borrow().eq(e.borrow())
                })
            })
        }
    }

    /// Sub-Expression Pattern
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct SubExpressionPattern<'p, P>(&'p P)
    where
        P: Expression;

    impl<'p, P> SubExpressionPattern<'p, P>
    where
        P: 'p + Expression,
    {
        pub(crate) fn new(pattern: &'p P) -> Self {
            Self(pattern)
        }

        fn matches_atom<E>(pattern: &P, atom: &E::Atom) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.cases().atom() {
                Some(pattern_atom) => pattern_atom == atom,
                _ => false,
            }
        }

        fn matches_group<E>(
            pattern: &P,
            group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>,
        ) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.cases().group() {
                Some(pattern_group) => {
                    group
                        .iter()
                        .any(move |e| Self::matches(pattern, e.borrow()))
                        || eq_by(pattern_group.iter(), group.iter(), move |p, e| {
                            p.borrow().eq(e.borrow())
                        })
                }
                _ => group
                    .iter()
                    .any(move |e| Self::matches(pattern, e.borrow())),
            }
        }

        #[inline]
        fn matches<E>(pattern: &P, expr: &E) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match expr.cases() {
                ExprRef::Atom(atom) => Self::matches_atom::<E>(pattern, atom),
                ExprRef::Group(group) => Self::matches_group::<E>(pattern, group),
            }
        }
    }

    impl<'p, P, E> Pattern<E> for SubExpressionPattern<'p, P>
    where
        E: Expression,
        P: Expression,
        P::Atom: PartialEq<E::Atom>,
    {
        #[inline]
        fn matches_atom(&self, atom: &E::Atom) -> bool {
            Self::matches_atom::<E>(&self.0, atom)
        }

        #[inline]
        fn matches_group(&self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool {
            Self::matches_group::<E>(&self.0, group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(&self.0, expr)
        }
    }

    /// Wild Card Pattern
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct WildCardPattern<'p, W, P>(W, &'p P)
    where
        P: Expression,
        W: FnMut(&P::Atom) -> bool;

    impl<'p, W, P> WildCardPattern<'p, W, P>
    where
        P: 'p + Expression,
        W: FnMut(&P::Atom) -> bool,
    {
        pub(crate) fn new(is_wildcard: W, pattern: &'p P) -> Self {
            Self(is_wildcard, pattern)
        }

        fn matches_atom<F, E>(is_wildcard: F, pattern: &P, atom: &E::Atom) -> bool
        where
            F: FnOnce(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.cases().atom() {
                Some(pattern_atom) => is_wildcard(pattern_atom) || pattern_atom == atom,
                _ => false,
            }
        }

        fn matches_group<F, E>(
            mut is_wildcard: F,
            pattern: &P,
            group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>,
        ) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.cases() {
                ExprRef::Atom(pattern_atom) if is_wildcard(pattern_atom) => true,
                ExprRef::Atom(_) => group
                    .iter()
                    .any(move |e| Self::matches(&mut is_wildcard, pattern, e.borrow())),
                ExprRef::Group(pattern_group) => {
                    group
                        .iter()
                        .any(|e| Self::matches(&mut is_wildcard, pattern, e.borrow()))
                        || eq_by(pattern_group.iter(), group.iter(), |p, e| {
                            Self::wildcard_equality(&mut is_wildcard, p.borrow(), e.borrow())
                        })
                }
            }
        }

        #[inline]
        fn matches<F, E>(is_wildcard: F, pattern: &P, expr: &E) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match expr.cases() {
                ExprRef::Atom(atom) => Self::matches_atom::<_, E>(is_wildcard, pattern, atom),
                ExprRef::Group(group) => Self::matches_group::<_, E>(is_wildcard, pattern, group),
            }
        }

        fn wildcard_equality<F, E>(is_wildcard: &mut F, pattern: &P, expr: &E) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.cases() {
                ExprRef::Atom(pattern_atom) => {
                    is_wildcard(pattern_atom)
                        || expr
                            .cases()
                            .atom()
                            .map_or(false, move |a| pattern_atom == a)
                }
                ExprRef::Group(pattern_group) => match expr.cases().group() {
                    Some(group) => eq_by(pattern_group.iter(), group.iter(), move |p, e| {
                        Self::wildcard_equality(is_wildcard, p.borrow(), e.borrow())
                    }),
                    _ => false,
                },
            }
        }
    }

    impl<'p, W, P, E> Pattern<E> for WildCardPattern<'p, W, P>
    where
        E: Expression,
        W: Fn(&P::Atom) -> bool,
        P: Expression,
        P::Atom: PartialEq<E::Atom>,
    {
        #[inline]
        fn matches_atom(&self, atom: &E::Atom) -> bool {
            Self::matches_atom::<_, E>(&self.0, &self.1, atom)
        }

        #[inline]
        fn matches_group(&self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool {
            Self::matches_group::<_, E>(&self.0, &self.1, group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(&self.0, &self.1, expr)
        }
    }

    impl<'p, W, P, E> PatternMut<E> for WildCardPattern<'p, W, P>
    where
        E: Expression,
        W: FnMut(&P::Atom) -> bool,
        P: Expression,
        P::Atom: PartialEq<E::Atom>,
    {
        #[inline]
        fn matches_atom(&mut self, atom: &E::Atom) -> bool {
            Self::matches_atom::<_, E>(&mut self.0, &self.1, atom)
        }

        #[inline]
        fn matches_group(&mut self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool {
            Self::matches_group::<_, E>(&mut self.0, &self.1, group)
        }

        #[inline]
        fn matches(&mut self, expr: &E) -> bool {
            Self::matches(&mut self.0, &self.1, expr)
        }
    }

    /// Basic Expression Shape
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum BasicShape {
        /// Atomic Shape
        Atom,

        /// Grouped Shape
        Group,

        /// Any Expression Shape
        Expr,
    }

    impl Default for BasicShape {
        #[inline]
        fn default() -> Self {
            Self::Expr
        }
    }

    impl BasicShape {
        /// Checks if the shape would match an atom.
        #[inline]
        pub fn matches_atom(&self) -> bool {
            *self == Self::Expr || *self == Self::Atom
        }

        /// Checks if the shape would match a group.
        #[inline]
        pub fn matches_group(&self) -> bool {
            *self == Self::Expr || *self == Self::Group
        }
    }

    /// Pattern over `BasicShape` Expression.
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct BasicShapePattern<'p, P>(&'p P)
    where
        P: Expression<Atom = BasicShape>;

    impl<'p, P> BasicShapePattern<'p, P>
    where
        P: 'p + Expression<Atom = BasicShape>,
    {
        pub(crate) fn new(pattern: &'p P) -> Self {
            Self(pattern)
        }

        fn matches_atom<E>(pattern: &P, atom: &E::Atom) -> bool
        where
            E: Expression,
        {
            let _ = atom;
            pattern
                .cases()
                .atom()
                .map_or(false, BasicShape::matches_atom)
        }

        fn matches_group<E>(
            pattern: &P,
            group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>,
        ) -> bool
        where
            E: Expression,
        {
            match pattern.cases() {
                ExprRef::Atom(pattern_atom) => pattern_atom.matches_group(),
                ExprRef::Group(pattern_group) => {
                    eq_by(pattern_group.iter(), group.iter(), move |p, e| {
                        Self::matches(p.borrow(), e.borrow())
                    })
                }
            }
        }

        #[inline]
        fn matches<E>(pattern: &P, expr: &E) -> bool
        where
            E: Expression,
        {
            match expr.cases() {
                ExprRef::Atom(atom) => Self::matches_atom::<E>(pattern, atom),
                ExprRef::Group(group) => Self::matches_group::<E>(pattern, group),
            }
        }
    }

    impl<'p, P, E> Pattern<E> for BasicShapePattern<'p, P>
    where
        E: Expression,
        P: Expression<Atom = BasicShape>,
    {
        #[inline]
        fn matches_atom(&self, atom: &E::Atom) -> bool {
            Self::matches_atom::<E>(&self.0, atom)
        }

        #[inline]
        fn matches_group(&self, group: <E::Group as IntoIteratorGen<E>>::IterGen<'_>) -> bool {
            Self::matches_group::<E>(&self.0, group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(&self.0, expr)
        }
    }
}

/// Iterator Module
pub mod iter {
    use core::{borrow::Borrow, slice};

    /// An `Iterator` generator that consumes by reference.
    pub trait IteratorGen<T> {
        type Item<'t>: Borrow<T>
        where
            T: 't;

        /// Underlying `Iterator` Type
        type Iter<'t>: Iterator<Item = Self::Item<'t>>
        where
            T: 't;

        /// Gets a new `Iterator`.
        fn iter(&self) -> Self::Iter<'_>;
    }

    /// Convert a type into a `IteratorGen`.
    pub trait IntoIteratorGen<T> {
        /// Underlying `IteratorGen` Type
        type IterGen<'t>: IteratorGen<T>
        where
            T: 't;

        /// Gets a new `IteratorGen`.
        fn gen(&self) -> Self::IterGen<'_>;
    }

    impl<T> IteratorGen<T> for &[T] {
        type Item<'t>
        where
            T: 't,
        = &'t T;

        type Iter<'t>
        where
            T: 't,
        = slice::Iter<'t, T>;

        #[inline]
        fn iter(&self) -> Self::Iter<'_> {
            (self[..]).iter()
        }
    }

    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    impl<T> IteratorGen<T> for &Vec<T> {
        type Item<'t>
        where
            T: 't,
        = &'t T;

        type Iter<'t>
        where
            T: 't,
        = slice::Iter<'t, T>;

        #[inline]
        fn iter(&self) -> Self::Iter<'_> {
            (self[..]).iter()
        }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    impl<T> IntoIteratorGen<T> for Vec<T> {
        type IterGen<'t>
        where
            T: 't,
        = &'t Self;

        #[inline]
        fn gen(&self) -> Self::IterGen<'_> {
            &self
        }
    }

    /* TODO: can we even implement this?
    /// Iterator Helper for use inside of `Expr`
    pub(crate) struct ExprIter<'e, E>
    where
        E: super::Expression,
    {
        iter: <<E::Group as IntoIteratorGen<E>>::IterGen<'e> as IteratorGen<E>>::Iter<'e>,
        phantom: PhantomData<&'e E>,
    }

    impl<'e, E> ExprIter<'e, E>
    where
        E: super::Expression,
    {
        pub(crate) fn from_container(container: &'e ExprIterContainer<'e, E>) -> Self {
            Self {
                iter: container.iter.iter(),
                phantom: PhantomData,
            }
        }

        pub(crate) fn inner_next(
            &mut self,
        ) -> Option<<<E::Group as IntoIteratorGen<E>>::IterGen<'e> as IteratorGen<E>>::Item<'e>>
        {
            self.iter.next()
        }
    }

    /// Container for an `ExprIter`
    pub(crate) struct ExprIterContainer<'e, E>
    where
        E: super::Expression,
    {
        iter: <E::Group as IntoIteratorGen<E>>::IterGen<'e>,
        phantom: PhantomData<&'e E>,
    }

    impl<'e, E> ExprIterContainer<'e, E>
    where
        E: super::Expression,
    {
        pub(crate) fn new(iter: <E::Group as IntoIteratorGen<E>>::IterGen<'e>) -> Self {
            Self {
                iter,
                phantom: PhantomData,
            }
        }
    }
    */

    /// Checks if two iterators are equal pointwise.
    pub fn eq_by<L, R, F>(lhs: L, rhs: R, mut eq: F) -> bool
    where
        L: IntoIterator,
        R: IntoIterator,
        F: FnMut(L::Item, R::Item) -> bool,
    {
        // TODO: when the nightly `iter_order_by` (issue #64295) is resolved,
        // switch to that and remove this function.
        let mut lhs = lhs.into_iter();
        let mut rhs = rhs.into_iter();
        loop {
            let x = match lhs.next() {
                None => return rhs.next().is_none(),
                Some(val) => val,
            };
            let y = match rhs.next() {
                None => return false,
                Some(val) => val,
            };
            if !eq(x, y) {
                return false;
            }
        }
    }
}

/// Vector Expressions
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod vec {
    use {
        super::*,
        alloc::{slice, string::String, vec::Vec},
    };

    /// Vector Expression Type over `String`s
    pub type StringExpr = Expr<String>;

    /// Vector Expression Type
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum Expr<A = ()> {
        /// Atomic expression
        Atom(A),

        /// Grouped expression
        Group(Vec<Self>),
    }

    impl<A> Expression for Expr<A> {
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

    impl<A> From<Expr<A>> for super::Expr<Expr<A>> {
        #[inline]
        fn from(expr: Expr<A>) -> Self {
            match expr {
                Expr::Atom(atom) => Self::Atom(atom),
                Expr::Group(group) => Self::Group(group),
            }
        }
    }

    impl<A> Default for Expr<A> {
        #[inline]
        fn default() -> Self {
            <Self as Expression>::default()
        }
    }

    #[cfg(feature = "parse")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
    impl<A> FromStr for Expr<A>
    where
        A: FromIterator<char>,
    {
        type Err = parse::Error;

        #[inline]
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Expression::from_str(s)
        }
    }

    /// Vector `MultiExpression` over `String`s
    pub type StringMultiExpr<G = ()> = MultiExpr<String, G>;

    /// Vector `MultiExpression` Type
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum MultiExpr<A = (), G = ()> {
        /// Atomic Expression
        Atom(A),

        /// Grouped Expression
        Group(Vec<Self>, G),
    }

    impl<A, G> Expression for MultiExpr<A, G> {
        type Atom = A;

        type Group = (Vec<Self>, G);

        #[inline]
        fn cases(&self) -> ExprRef<Self> {
            match self {
                Self::Atom(atom) => ExprRef::Atom(atom),
                Self::Group(group, group_type) => ExprRef::Group((group, group_type)),
            }
        }

        #[inline]
        fn from_atom(atom: <Self as Expression>::Atom) -> Self {
            Self::Atom(atom)
        }

        #[inline]
        fn from_group(group: <Self as Expression>::Group) -> Self {
            Self::Group(group.0, group.1)
        }
    }

    impl<A, G> IteratorGen<MultiExpr<A, G>> for (&Vec<MultiExpr<A, G>>, &G) {
        type Item<'t>
        where
            A: 't,
            G: 't,
        = &'t MultiExpr<A, G>;

        type Iter<'t>
        where
            A: 't,
            G: 't,
        = slice::Iter<'t, MultiExpr<A, G>>;

        #[inline]
        fn iter(&self) -> Self::Iter<'_> {
            self.0.iter()
        }
    }

    impl<A, G> IntoIteratorGen<MultiExpr<A, G>> for (Vec<MultiExpr<A, G>>, G) {
        type IterGen<'t>
        where
            A: 't,
            G: 't,
        = (&'t Vec<MultiExpr<A, G>>, &'t G);

        fn gen(&self) -> Self::IterGen<'_> {
            (&self.0, &self.1)
        }
    }

    impl<A, G> From<MultiExpr<A, G>> for super::Expr<MultiExpr<A, G>> {
        #[inline]
        fn from(expr: MultiExpr<A, G>) -> Self {
            match expr {
                MultiExpr::Atom(atom) => Self::Atom(atom),
                MultiExpr::Group(group, group_type) => Self::Group((group, group_type)),
            }
        }
    }

    impl<A, G> FromIterator<MultiExpr<A, G>> for (Vec<MultiExpr<A, G>>, G)
    where
        G: Default,
    {
        fn from_iter<I>(iter: I) -> Self
        where
            I: IntoIterator<Item = MultiExpr<A, G>>,
        {
            (iter.into_iter().collect(), Default::default())
        }
    }

    impl<A, G> Default for MultiExpr<A, G>
    where
        G: Default,
    {
        #[inline]
        fn default() -> Self {
            <Self as Expression>::default()
        }
    }

    impl<'e, A, G> HasGroupType<'e, MultiExpr<A, G>, G>
        for <<MultiExpr<A, G> as Expression>::Group as IntoIteratorGen<MultiExpr<A, G>>>::IterGen<
            'e,
        >
    {
        #[inline]
        fn group_type(&self) -> &'e G {
            self.1
        }
    }
}

/// Buffered Expressions
#[cfg(feature = "buffered")]
#[cfg_attr(docsrs, doc(cfg(feature = "buffered")))]
pub mod buffered {
    use {super::*, alloc::vec::Vec};

    /// Buffered Expression Type
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct Expr<T, LengthIndex = usize, ShapeIndex = usize> {
        atoms: Vec<T>,
        lengths: Vec<LengthIndex>,
        shape: Vec<ShapeIndex>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExprGroup<T> {
        inner: Expr<T>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExprView<'t, T> {
        base: &'t Expr<T>,
        index: usize,
    }

    impl<'t, T> Borrow<Expr<T>> for ExprView<'t, T> {
        fn borrow(&self) -> &Expr<T> {
            todo!()
        }
    }

    pub struct ExprViewIterator<'t, T> {
        _base: &'t Expr<T>,
        index: usize,
    }

    impl<'t, T> ExprViewIterator<'t, T> {
        fn new(base: &'t Expr<T>) -> Self {
            Self {
                _base: base,
                index: 0,
            }
        }
    }

    impl<'t, T> Iterator for ExprViewIterator<'t, T> {
        type Item = ExprView<'t, T>;

        fn next(&mut self) -> Option<Self::Item> {
            self.index += 1;
            todo!()
        }
    }

    impl<T> IteratorGen<Expr<T>> for &Expr<T> {
        type Item<'t>
        where
            T: 't,
        = ExprView<'t, T>;

        type Iter<'t>
        where
            T: 't,
        = ExprViewIterator<'t, T>;

        fn iter(&self) -> Self::Iter<'_> {
            ExprViewIterator::new(self)
        }
    }

    impl<T> IntoIteratorGen<Expr<T>> for ExprGroup<T> {
        type IterGen<'t>
        where
            T: 't,
        = &'t Expr<T>;

        #[inline]
        fn gen(&self) -> Self::IterGen<'_> {
            &self.inner
        }
    }

    impl<T> Expression for Expr<T> {
        type Atom = Vec<T>;

        type Group = ExprGroup<T>;

        fn cases(&self) -> ExprRef<Self> {
            todo!()
        }

        fn from_atom(atom: <Self as Expression>::Atom) -> Self {
            Self {
                atoms: atom,
                lengths: Vec::default(),
                shape: Vec::default(),
            }
        }

        fn from_group(group: <Self as Expression>::Group) -> Self {
            group.inner
        }
    }

    impl<T> From<Expr<T>> for super::Expr<Expr<T>> {
        #[inline]
        fn from(expr: Expr<T>) -> Self {
            let _ = expr;
            todo!()
        }
    }
}
