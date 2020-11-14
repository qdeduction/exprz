// file: core/src/lib.rs
// authors: Brandon H. Gomes

//! ExprZ Core Library

#![no_std]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]

use {crate::iter::*, core::iter::FromIterator};

/// Expression Tree
pub trait Expression
where
    Self: Into<Expr<Self>>,
{
    /// Atomic element type
    type Atom;

    /// Group expression type
    type Group: IntoIteratorGen<Self>;

    /// Get a reference to the underlying `Expression` type.
    fn cases(&self) -> ExprRef<Self>;

    /// Build an `Expression` from an atomic element.
    fn from_atom(atom: Self::Atom) -> Self;

    /// Build an `Expression` from a grouped expression.
    fn from_group(group: Self::Group) -> Self;

    /// Convert from the [canonical enumeration].
    ///
    /// [canonical enumeration]: enum.Expr.html
    #[inline]
    fn from_expr(expr: Expr<Self>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::from_atom(atom),
            Expr::Group(group) => Self::from_group(group),
        }
    }

    /// Check if the `Expression` is atomic.
    #[must_use]
    #[inline]
    fn is_atom(&self) -> bool {
        self.cases().is_atom()
    }

    /// Check if the `Expression` is a grouped expression.
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
    #[inline]
    #[track_caller]
    fn unwrap_atom(self) -> Self::Atom {
        self.into().unwrap_atom()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    #[inline]
    #[track_caller]
    fn unwrap_group(self) -> Self::Group {
        self.into().unwrap_group()
    }

    /// Clone an `Expression` that has `Clone`-able `Atom`s.
    #[inline]
    fn clone(&self) -> Self
    where
        Self::Atom: Clone,
        Self::Group: FromIterator<Self>,
    {
        Self::from_expr(self.cases().into())
    }

    /// Check if two `Expression`s are equal using `PartialEq` on their `Atom`s.
    fn eq<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
    {
        match (self.cases(), other.cases()) {
            (ExprRef::Atom(lhs), ExprRef::Atom(rhs)) => lhs == rhs,
            (ExprRef::Group(lhs), ExprRef::Group(rhs)) => {
                eq_by(lhs.iter(), rhs.iter(), move |l, r| l.eq(&r))
            }
            _ => false,
        }
    }

    /// Check if an `Expression` is a sub-tree of another `Expression` using `PartialEq` on their
    /// `Atom`s.
    fn is_subexpression<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
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
                        || eq_by(group.iter(), other.iter(), move |l, r| l.eq(&r))
                }
            },
        }
    }

    /// Extend a function on `Atom`s to a function on `Expression`s.
    fn map<E, F>(self, f: &mut F) -> E
    where
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(Self::Atom) -> E::Atom,
    {
        match self.into() {
            Expr::Atom(atom) => E::from_atom(f(atom)),
            Expr::Group(group) => {
                E::from_group(group.gen().iter().map(move |e| e.map(f)).collect())
            }
        }
    }

    /// Extend a function on `&Atom`s to a function on `&Expression`s.
    fn map_ref<E, F>(&self, f: &mut F) -> E
    where
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(&Self::Atom) -> E::Atom,
    {
        match self.cases() {
            ExprRef::Atom(atom) => E::from_atom(f(atom)),
            ExprRef::Group(group) => {
                E::from_group(group.iter().map(move |e| e.map_ref(f)).collect())
            }
        }
    }

    /// Substitute an `Expression` into each `Atom` of `self`.
    fn substitute<F>(self, f: &mut F) -> Self
    where
        Self::Group: FromIterator<Self>,
        F: FnMut(Self::Atom) -> Self,
    {
        match self.into() {
            Expr::Atom(atom) => f(atom),
            Expr::Group(group) => {
                Self::from_group(group.gen().iter().map(move |e| e.substitute(f)).collect())
            }
        }
    }

    /// Use an iterator generator as a piecewise function to substitute an `Expression` into each
    /// `Atom` of `self`.
    fn substitute_from_iter<'s, I>(self, iter: &I) -> Self
    where
        Self::Atom: 's + PartialEq,
        Self::Group: FromIterator<Self>,
        I: IteratorGen<(&'s Self::Atom, Self)>,
    {
        self.substitute(&mut move |atom| {
            util::piecewise_map(&atom, iter.iter()).unwrap_or_else(move || Self::from_atom(atom))
        })
    }

    /// Substitute an `Expression` into each `Atom` of `&self`.
    fn substitute_ref<F>(&self, f: &mut F) -> Self
    where
        Self::Group: FromIterator<Self>,
        F: FnMut(&Self::Atom) -> Self,
    {
        match self.cases() {
            ExprRef::Atom(atom) => f(atom),
            ExprRef::Group(group) => {
                Self::from_group(group.iter().map(move |e| e.substitute_ref(f)).collect())
            }
        }
    }

    /// Use an iterator generator as a piecewise function to substitute an `Expression` into each
    /// `Atom` of `&self`.
    fn substitute_ref_from_iter<'s, I>(&self, iter: &I) -> Self
    where
        Self: 's,
        Self::Atom: PartialEq + Clone,
        Self::Group: FromIterator<Self>,
        I: IteratorGen<(&'s Self::Atom, &'s Self)>,
    {
        self.substitute_ref(&mut move |atom| {
            util::piecewise_map(atom, iter.iter())
                .map_or_else(move || Self::from_atom(atom.clone()), Expression::clone)
        })
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
    /// Check if the `ExprRef` is atomic.
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, ExprRef::Atom(_))
    }

    /// Check if the `ExprRef` is a grouped expression `IteratorGen`.
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, ExprRef::Group(_))
    }

    /// Converts from an `ExprRef<E>` to an `Option<&E::Atom>`.
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<&'e E::Atom> {
        match self {
            ExprRef::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an `ExprRef<E>` to an `Option<E::Group::IterGen>`.
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<<E::Group as IntoIteratorGen<E>>::IterGen<'e>> {
        match self {
            ExprRef::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Returns the contained `Atom` value, consuming the `self` value.
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> &'e E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> <E::Group as IntoIteratorGen<E>>::IterGen<'e> {
        self.group().unwrap()
    }
}

/// Canonical Concrete `Expression` Type
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

impl<E> Expr<E>
where
    E: Expression,
{
    /// Check if the `Expr` is atomic.
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Expr::Atom(_))
    }

    /// Check if the `Expr` is a grouped expression.
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
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained `Group` value, consuming the `self` value.
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> E::Group {
        self.group().unwrap()
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
        self.inner_next().map(E::into)
    }
}

///
///
pub mod parse {
    use {
        super::Expression,
        core::{
            iter::{from_fn, FromIterator, Peekable},
            result,
        },
    };

    /// `Expression` Parsing Error
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum Error {
        /// Multiple expressions at top level
        MultiExpr,

        /// No closing quote
        MissingQuote,

        /// Group was not closed
        OpenGroup,

        /// Group was not opened
        UnopenedGroup,
    }

    /// `Expression` Parsing Result Type
    pub type Result<T> = result::Result<T, Error>;

    /// Meaningful Symbols for `Expression` Parsing
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum SymbolType {
        /// Whitespace
        Whitespace,

        /// Start of a group
        GroupOpen,

        /// End of a group
        GroupClose,

        /// Start of a quoted sub-string
        Quote,

        /// Other characters
        Other,
    }

    impl SymbolType {
        /// Checks if the classified symbol is whitespace.
        #[inline]
        pub fn is_whitespace<T, C>(classify: C) -> impl Fn(&T) -> bool
        where
            C: Fn(&T) -> SymbolType,
        {
            move |t| classify(t) == SymbolType::Whitespace
        }

        /// Checks if the classified symbol is not whitespace.
        #[inline]
        pub fn is_not_whitespace<T, C>(classify: C) -> impl Fn(&T) -> bool
        where
            C: Fn(&T) -> SymbolType,
        {
            move |t| classify(t) != SymbolType::Whitespace
        }
    }

    /// Parse an `Expression` from an `Iterator` over `collect`-able symbols.
    pub fn parse<T, C, I, E>(classify: C, iter: I) -> Result<E>
    where
        C: Fn(&T) -> SymbolType,
        I: IntoIterator<Item = T>,
        E: Expression,
        E::Atom: FromIterator<T>,
        E::Group: FromIterator<E>,
    {
        let mut stripped = iter
            .into_iter()
            .skip_while(SymbolType::is_whitespace(&classify))
            .peekable();
        match stripped.peek() {
            Some(peek) => match classify(&peek) {
                SymbolType::GroupOpen => parse_group_continue(&classify, &mut stripped),
                SymbolType::GroupClose => Err(Error::UnopenedGroup),
                _ => {
                    let atom = parse_atom_continue(&classify, &mut stripped)?;
                    if let Some(next) = stripped.next() {
                        match classify(&next) {
                            SymbolType::Whitespace => {
                                if stripped.any(|t| SymbolType::is_not_whitespace(&classify)(&t)) {
                                    return Err(Error::MultiExpr);
                                }
                            }
                            SymbolType::GroupOpen | SymbolType::GroupClose => {
                                return Err(Error::MultiExpr);
                            }
                            _ => {}
                        }
                    }
                    Ok(E::from_atom(atom))
                }
            },
            _ => Ok(E::from_atom(E::Atom::from_iter(None))),
        }
    }

    /// Parse a `Group` from an `Iterator` over `collect`-able symbols.
    #[inline]
    pub fn parse_group<T, C, I, E>(classify: C, iter: I) -> Result<E>
    where
        C: Fn(&T) -> SymbolType,
        I: IntoIterator<Item = T>,
        E: Expression,
        E::Atom: FromIterator<T>,
        E::Group: FromIterator<E>,
    {
        parse_group_continue(classify, &mut iter.into_iter().peekable())
    }

    #[inline]
    fn parse_group_continue<T, C, I, E>(classify: C, iter: &mut Peekable<I>) -> Result<E>
    where
        C: Fn(&T) -> SymbolType,
        I: Iterator<Item = T>,
        E: Expression,
        E::Atom: FromIterator<T>,
        E::Group: FromIterator<E>,
    {
        parse_group_continue_at_depth(0, classify, iter)
    }

    fn parse_group_continue_at_depth<T, C, I, E>(
        depth: usize,
        classify: C,
        iter: &mut Peekable<I>,
    ) -> Result<E>
    where
        C: Fn(&T) -> SymbolType,
        I: Iterator<Item = T>,
        E: Expression,
        E::Atom: FromIterator<T>,
        E::Group: FromIterator<E>,
    {
        // FIXME[check]: might need to `.fuse()` after the `.filter_map(...)` to ensure we are
        // stopping at `GroupClose`.
        //
        // FIXME[check]: could use `loop { match iter.peek() { ... } }` instead of `filter_map`
        // technique to skip over whitespace.
        //
        let target: Result<_> = from_fn(|| match iter.peek() {
            Some(peek) => match classify(&peek) {
                SymbolType::Whitespace => Some(None),
                SymbolType::GroupOpen => Some(Some(parse_group_continue_at_depth(
                    depth + 1,
                    &classify,
                    iter,
                ))),
                SymbolType::GroupClose => None,
                _ => Some(Some(parse_atom_continue(&classify, iter).map(E::from_atom))),
            },
            _ => Some(Some(Err(Error::OpenGroup))),
        })
        .filter_map(move |t| t)
        .collect();
        if depth == 0 && iter.find(SymbolType::is_not_whitespace(classify)).is_some() {
            Err(Error::MultiExpr)
        } else {
            target.map(E::from_group)
        }
    }

    /// Parse an `Atom` from an `Iterator` over `collect`-able symbols.
    #[inline]
    pub fn parse_atom<T, C, I, A>(classify: C, iter: I) -> Result<A>
    where
        C: Fn(&T) -> SymbolType,
        I: IntoIterator<Item = T>,
        A: FromIterator<T>,
    {
        parse_atom_continue(classify, &mut iter.into_iter())
    }

    fn parse_atom_continue<T, C, I, A>(classify: C, iter: &mut I) -> Result<A>
    where
        C: Fn(&T) -> SymbolType,
        I: Iterator<Item = T>,
        A: FromIterator<T>,
    {
        let mut inside_quote = false;
        let atom = iter
            .take_while(move |t| {
                if inside_quote {
                    if classify(&t) == SymbolType::Quote {
                        inside_quote = false;
                    }
                } else {
                    match classify(&t) {
                        SymbolType::Quote => inside_quote = true,
                        SymbolType::Other => {}
                        _ => return false,
                    }
                }
                true
            })
            .collect();
        if inside_quote {
            Err(Error::MissingQuote)
        } else {
            Ok(atom)
        }
    }

    /// Default classification for the `char` type.
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

    /// Parse a string-like `Expression` from an iterator over characters.
    #[inline]
    pub fn from_chars<I, E>(iter: I) -> Result<E>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        parse(default_char_classification, iter)
    }

    /// Parse a string-like `Expression` from a string.
    #[inline]
    pub fn from_str<S, E>(s: S) -> Result<E>
    where
        S: AsRef<str>,
        E: Expression,
        E::Group: FromIterator<E>,
        E::Atom: FromIterator<char>,
    {
        from_chars(s.as_ref().chars())
    }

    /// Parse a string-like expression `Group` from an iterator over characters.
    #[inline]
    pub fn from_chars_as_group<I, E>(iter: I) -> Result<E::Group>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: use `parse_group` instead of chaining
        from_chars(Some('(').into_iter().chain(iter).chain(Some(')')))
            .map(move |e: E| e.unwrap_group())
    }

    /// Parse a string-like expression `Group` from a string.
    #[inline]
    pub fn from_str_as_group<S, E>(s: S) -> Result<E::Group>
    where
        S: AsRef<str>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        from_chars_as_group::<_, E>(s.as_ref().chars())
    }
}

/// Iterator Module
pub mod iter {
    use core::marker::PhantomData;

    /// An `Iterator` generator that consumes by reference.
    pub trait IteratorGen<T> {
        /// Underlying `Iterator` Type
        type Iter<'t>: Iterator<Item = T>
        where
            T: 't;

        /// Get a new `Iterator`.
        fn iter(&self) -> Self::Iter<'_>;
    }

    /// Convert a type into a `IteratorGen`.
    pub trait IntoIteratorGen<T> {
        /// Underlying `IteratorGen` Type
        type IterGen<'t>: IteratorGen<T>
        where
            T: 't;

        /// Get a new `IteratorGen`.
        fn gen(&self) -> Self::IterGen<'_>;
    }

    /// Iterator Helper for use inside of `Expr`
    pub struct ExprIter<'e, E>
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

        pub(crate) fn inner_next(&mut self) -> Option<E> {
            self.iter.next()
        }
    }

    /// Container for an `ExprIter`
    pub struct ExprIterContainer<'e, E>
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

    // TODO: when the nightly `iter_order_by` (issue #64295) is resolved,
    // switch to that and remove this function.
    pub(crate) fn eq_by<L, R, F>(lhs: L, rhs: R, mut eq: F) -> bool
    where
        L: IntoIterator,
        R: IntoIterator,
        F: FnMut(L::Item, R::Item) -> bool,
    {
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

/// Utilities Module
pub mod util {
    /// Turn an `Iterator` over pairs into a piecewise function.
    #[inline]
    pub fn piecewise_map<T, A, B, I>(t: T, iter: I) -> Option<B>
    where
        T: PartialEq<A>,
        I: IntoIterator<Item = (A, B)>,
    {
        // TODO: replace `find_map` body with `(t == a).then_some(b)` from nightly
        iter.into_iter()
            .find_map(move |(a, b)| if t == a { Some(b) } else { None })
    }
}
