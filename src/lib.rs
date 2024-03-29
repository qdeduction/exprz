//! An Expression Library

// TODO:  implement `Deref/Borrow/ToOwned` traits where possible
// TODO:  implement `std::error::Error` for errors if `std` is enabled
// TODO:  add derive macros for `E: Expression` to get `Clone`, `PartialEq`, ... etc. for free
// FIXME: we need to put all of the implementations in the main traits and have `Expr/ExprRef` use
//        the trait implementations somehow
// FIXME: for `Eq` implementations we need to constrain `E::Atom` to `Eq` not `PartialEq`
// TODO:  add async parsing
// TODO:  reconsider how we use `Clone`/`PartialEq` on interfaces, maybe we should just have `E: Trait`
// TODO:  rename `is_subexpression` to `contains`? (and flip the arguments) add `contains_atom/group`?

#![cfg_attr(docsrs, feature(doc_cfg), forbid(broken_intra_doc_links))]
#![feature(associated_type_defaults, generic_associated_types)]
#![allow(incomplete_features)]
#![forbid(missing_docs)]
#![forbid(unsafe_code)]
#![no_std]

use core::{iter::FromIterator, slice};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "parse")]
use core::str::FromStr;

#[cfg(feature = "rayon")]
use rayon::iter::{
    FromParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

#[cfg(feature = "serde")]
use serde::{Serialize, Serializer};

#[cfg(all(feature = "parse", feature = "serde"))]
use serde::{Deserialize, Deserializer};

/// Package Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// [`Expression`] Reference Trait
pub trait Reference<'e, E>: From<&'e E>
where
    E: 'e + Expression,
{
    /// Returns the inner expression reference.
    fn cases(self) -> ExprRef<'e, E>;

    /// Returns the kind of an expression.
    #[must_use]
    #[inline]
    fn kind(self) -> ExprKind {
        ExprRef::kind(&self.cases())
    }

    /// Checks if the [`Reference`] is an atomic expression [`&E::Atom`](Expression::Atom).
    #[allow(clippy::wrong_self_convention)]
    #[must_use]
    #[inline]
    fn is_atom(self) -> bool
    where
        Self: Sized,
    {
        ExprRef::is_atom(&self.cases())
    }

    /// Checks if the [`Reference`] is a grouped expression [`Group<E>::Ref`].
    #[allow(clippy::wrong_self_convention)]
    #[must_use]
    #[inline]
    fn is_group(self) -> bool
    where
        Self: Sized,
    {
        ExprRef::is_group(&self.cases())
    }

    /// Converts from a [`Reference<E>`](Reference) to an [`Option`]`<`[`&E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    fn atom(self) -> Option<&'e E::Atom>
    where
        Self: Sized,
    {
        ExprRef::atom(self.cases())
    }

    /// Converts from a [`Reference<E>`](Reference) to an [`Option`]`<`[`GroupRef<E>`]`>`.
    #[must_use]
    #[inline]
    fn group(self) -> Option<GroupRef<'e, E>>
    where
        Self: Sized,
    {
        ExprRef::group(self.cases())
    }

    /// Returns the contained [`Atom`](Expression::Atom) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a [`Group`](Expression::Group).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    fn unwrap_atom(self) -> &'e E::Atom
    where
        Self: Sized,
    {
        ExprRef::unwrap_atom(self.cases())
    }

    /// Returns the contained [`Group`](Expression::Group) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an [`Atom`](Expression::Atom).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    fn unwrap_group(self) -> GroupRef<'e, E>
    where
        Self: Sized,
    {
        ExprRef::unwrap_group(self.cases())
    }

    /// Returns new owned copy of the underlying [`Expression`].
    #[allow(clippy::wrong_self_convention)]
    #[inline]
    fn to_owned(self) -> E
    where
        Self: Sized,
        E::Atom: Clone,
        E::Group: FromIterator<E>,
    {
        ExprRef::to_owned(self.cases())
    }

    /// Performs substitution over the [`Expression`] by reference.
    #[inline]
    fn substitute_ref<F>(self, f: F) -> E
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        ExprRef::substitute_ref(&self.cases(), f)
    }
}

impl<'e, E> Reference<'e, E> for &'e E
where
    E: Expression,
{
    #[inline]
    fn cases(self) -> ExprRef<'e, E> {
        self.cases()
    }
}

impl<'e, E> Reference<'e, E> for ExprRef<'e, E>
where
    E: Expression,
{
    #[inline]
    fn cases(self) -> Self {
        self
    }
}

/// [`Expression`] Mutable Reference Trait
pub trait ReferenceMut<'e, E>: From<&'e mut E>
where
    E: 'e + Expression,
    E::Group: GroupMut<E>,
{
    /// Returns the inner expression mutable reference.
    fn cases_mut(self) -> ExprRefMut<'e, E>;
}

impl<'e, E> ReferenceMut<'e, E> for &'e mut E
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    #[inline]
    fn cases_mut(self) -> ExprRefMut<'e, E> {
        // FIXME: only way to implement this is to ask `E: Expression` for it
        todo!()
    }
}

impl<'e, E> ReferenceMut<'e, E> for ExprRefMut<'e, E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    #[inline]
    fn cases_mut(self) -> Self {
        self
    }
}

/// [`Expression Group`](Expression::Group) Reference Trait
pub trait GroupReference<E>
where
    E: Expression,
{
    /// Element of a [`GroupReference`]
    type Item<'e>: Reference<'e, E>
    where
        E: 'e,
    = &'e E;

    /// Iterator over [`GroupReference::Item`]
    type Iter<'e>: Iterator<Item = Self::Item<'e>>
    where
        E: 'e;

    /// Returns a group reference iterator.
    fn iter(&self) -> Self::Iter<'_>;

    /// Returns a size hint for the underlying iterator.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter().size_hint()
    }

    /// Returns the length of the group reference if it is known exactly.
    #[inline]
    fn len(&self) -> Option<usize> {
        let (min, max) = self.size_hint();
        max.filter(move |m| *m == min)
    }

    /// Returns `true` if the length of the group reference is known to be exactly zero.
    #[inline]
    fn is_empty(&self) -> bool {
        matches!(self.len(), Some(0))
    }

    /// Returns a reference to an element at the given position.
    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item<'_>> {
        self.iter().nth(index)
    }

    /// Returns new owned [`Group`](Expression::Group) from a [`GroupReference`].
    #[inline]
    fn to_owned(&self) -> E::Group
    where
        E::Atom: Clone,
        E::Group: FromIterator<E>,
    {
        self.iter().map(Reference::to_owned).collect()
    }

    /// Performs substitution over the underlying [`Group`](Expression::Group) by reference.
    #[inline]
    fn substitute_ref<F>(&self, mut f: F) -> E::Group
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        ExprRef::substitute_ref_group_inner(self.iter(), &mut f)
    }
}

impl<E> GroupReference<E> for &[E]
where
    E: Expression,
{
    type Iter<'e>
    where
        E: 'e,
    = slice::Iter<'e, E>;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self[..].iter()
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<E> GroupReference<E> for &Vec<E>
where
    E: Expression,
{
    type Iter<'e>
    where
        E: 'e,
    = slice::Iter<'e, E>;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self[..].iter()
    }
}

/// Parallel [`Expression Group`](Expression::Group) Reference Trait
#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub trait ParallelGroupReference<E>: GroupReference<E>
where
    E: Expression,
{
    /// Parallel Iterator over [`GroupReference::Item`]
    type ParIter<'e>: ParallelIterator<Item = <Self as GroupReference<E>>::Item<'e>>
    where
        E: 'e;

    /// Returns a parallel group reference iterator.
    fn par_iter(&self) -> Self::ParIter<'_>;
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<E> ParallelGroupReference<E> for &[E]
where
    E: Expression + Sync,
{
    type ParIter<'e>
    where
        E: 'e,
    = rayon::slice::Iter<'e, E>;

    #[inline]
    fn par_iter(&self) -> Self::ParIter<'_> {
        self[..].par_iter()
    }
}

#[cfg(all(feature = "alloc", feature = "rayon"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "rayon"))))]
impl<E> ParallelGroupReference<E> for &Vec<E>
where
    E: Expression + Sync,
{
    type ParIter<'e>
    where
        E: 'e,
    = rayon::slice::Iter<'e, E>;

    #[inline]
    fn par_iter(&self) -> Self::ParIter<'_> {
        self[..].par_iter()
    }
}

/// Indexed Parallel [`Expression Group`](Expression::Group) Reference Trait
#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub trait IndexedParallelGroupReference<E>: GroupReference<E>
where
    E: Expression,
{
    /// Indexed Parallel Iterator over [`GroupReference::Item`]
    type IndexedParIter<'e>: IndexedParallelIterator<Item = <Self as GroupReference<E>>::Item<'e>>
    where
        E: 'e;

    /// Returns an indexed parallel group reference iterator.
    fn indexed_par_iter(&self) -> Self::IndexedParIter<'_>;
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<E> IndexedParallelGroupReference<E> for &[E]
where
    E: Expression + Sync,
{
    type IndexedParIter<'e>
    where
        E: 'e,
    = rayon::slice::Iter<'e, E>;

    #[inline]
    fn indexed_par_iter(&self) -> Self::IndexedParIter<'_> {
        self.par_iter()
    }
}

#[cfg(all(feature = "alloc", feature = "rayon"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "rayon"))))]
impl<E> IndexedParallelGroupReference<E> for &Vec<E>
where
    E: Expression + Sync,
{
    type IndexedParIter<'e>
    where
        E: 'e,
    = rayon::slice::Iter<'e, E>;

    #[inline]
    fn indexed_par_iter(&self) -> Self::IndexedParIter<'_> {
        self.par_iter()
    }
}

/* FIXME:
#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<E, I> ParallelGroupReference<E> for I
where
    E: Expression,
    I: IndexedParallelGroupReference<E>,
{
    type ParIter<'e>
    where
        E: 'e,
    = I::IndexedParIter<'e>;

    #[inline]
    fn par_iter(&self) -> Self::ParIter<'_> {
        self.indexed_par_iter()
    }
}
*/

/// [`Expression Group`](Expression::Group) Mutable Reference Trait
pub trait GroupReferenceMut<E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    /// Element of a [`GroupReferenceMut`]
    type ItemMut<'e>: ReferenceMut<'e, E>
    where
        E: 'e,
    = &'e mut E;

    /// Iterator over [`GroupReferenceMut::ItemMut`]
    type IterMut<'e>: Iterator<Item = Self::ItemMut<'e>>
    where
        E: 'e;

    /// Returns a group mutable reference iterator.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

impl<E> GroupReferenceMut<E> for &mut [E]
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    type IterMut<'e>
    where
        E: 'e,
    = slice::IterMut<'e, E>;

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self[..].iter_mut()
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<E> GroupReferenceMut<E> for &mut Vec<E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    type IterMut<'e>
    where
        E: 'e,
    = slice::IterMut<'e, E>;

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self[..].iter_mut()
    }
}

/// [`Expression Group`](Expression::Group) Trait
pub trait Group<E>
where
    E: Expression,
{
    /// [`Group`](Expression::Group) Reference Type
    type Ref<'e>: GroupReference<E>
    where
        E: 'e;

    /// Returns a shared reference to the group.
    fn as_ref(&self) -> Self::Ref<'_>;

    /// Returns a size hint for the underlying iterator.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_ref().size_hint()
    }

    /// Returns the length of the group if it is known exactly.
    #[inline]
    fn len(&self) -> Option<usize> {
        self.as_ref().len()
    }

    /// Returns `true` if the length of the group is known to be exactly zero.
    #[inline]
    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    /// Builds an empty group.
    #[inline]
    fn empty() -> E::Group
    where
        E::Group: FromIterator<E>,
    {
        E::Group::from_iter(None)
    }

    /// Returns a cloned expression group.
    #[inline]
    fn clone(&self) -> E::Group
    where
        E::Atom: Clone,
        E::Group: FromIterator<E>,
    {
        self.as_ref().to_owned()
    }

    /// Performs substitution over the [`Group`].
    #[inline]
    fn substitute<F>(self, mut f: F) -> E::Group
    where
        Self: Sized + IntoIterator<Item = E>,
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        Expr::substitute_group_inner(self.into_iter(), &mut f)
    }

    /// Performs substitution over the [`Group`] by reference.
    #[inline]
    fn substitute_ref<F>(&self, f: F) -> E::Group
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        self.as_ref().substitute_ref(f)
    }
}

impl<E> Group<E> for [E]
where
    E: Expression,
{
    type Ref<'e>
    where
        E: 'e,
    = &'e Self;

    #[inline]
    fn as_ref(&self) -> Self::Ref<'_> {
        self
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<E> Group<E> for Vec<E>
where
    E: Expression,
{
    type Ref<'e>
    where
        E: 'e,
    = &'e Self;

    #[inline]
    fn as_ref(&self) -> Self::Ref<'_> {
        self
    }
}

/// [`Expression`] [`Group Reference`](GroupReference) Alias
pub type GroupRef<'e, E> = <<E as Expression>::Group as Group<E>>::Ref<'e>;

/// [`Expression`] [`Group Reference Iterator`](GroupReference::Iter) Alias
pub type GroupRefIter<'e, 'i, E> = <GroupRef<'e, E> as GroupReference<E>>::Iter<'i>;

/// [`Expression`] [`Group Reference`](GroupReference) [`Iterator Item`](GroupReference::Item) Alias
pub type GroupRefItem<'e, 'i, E> = <GroupRef<'e, E> as GroupReference<E>>::Item<'i>;

/// Mutable [`Expression Group`](Expression::Group) Trait
pub trait GroupMut<E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    /// [`Group`](Expression::Group) Mutable Reference Type
    type RefMut<'e>: GroupReferenceMut<E>
    where
        E: 'e;

    /// Returns a mutable reference to the group.
    fn as_mut(&mut self) -> Self::RefMut<'_>;
}

impl<E> GroupMut<E> for [E]
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    type RefMut<'e>
    where
        E: 'e,
    = &'e mut Self;

    #[inline]
    fn as_mut(&mut self) -> Self::RefMut<'_> {
        self
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<E> GroupMut<E> for Vec<E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    type RefMut<'e>
    where
        E: 'e,
    = &'e mut Self;

    #[inline]
    fn as_mut(&mut self) -> Self::RefMut<'_> {
        self
    }
}

/// [`Expression`] [`Group Mutable Reference`](GroupReferenceMut) Alias
pub type GroupRefMut<'e, E> = <<E as Expression>::Group as GroupMut<E>>::RefMut<'e>;

/// [`Expression`] [`Group Mutable Reference Iterator`](GroupReferenceMut::IterMut) Alias
pub type GroupRefMutIter<'e, 'i, E> = <GroupRefMut<'e, E> as GroupReferenceMut<E>>::IterMut<'i>;

/// [`Expression`] [`Group Mutable Reference`](GroupReferenceMut) [`Iterator Item`](GroupReferenceMut::ItemMut) Alias
pub type GroupRefMutItem<'e, 'i, E> = <GroupRefMut<'e, E> as GroupReferenceMut<E>>::ItemMut<'i>;

/// Expression Trait
pub trait Expression
where
    Self: Into<Expr<Self>>,
{
    /// Atomic Element Type
    type Atom;

    /// [`Group`] Expression Type
    type Group: Group<Self>;

    /// Returns a reference to the underlying [`Expression`] type.
    fn cases(&self) -> ExprRef<Self>;

    /// Builds an [`Expression`] from an atomic element.
    fn from_atom(atom: Self::Atom) -> Self;

    /// Builds an [`Expression`] from a grouped expression.
    fn from_group(group: Self::Group) -> Self;

    /// Converts from the [canonical enumeration](Expr).
    #[must_use]
    #[inline]
    fn from_expr(expr: Expr<Self>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::from_atom(atom),
            Expr::Group(group) => Self::from_group(group),
        }
    }

    /// Parses a string into an [`Expression`].
    #[cfg(feature = "parse")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
    #[inline]
    fn from_str(s: &str) -> parse::Result<Self, parse::FromCharactersError>
    where
        Self::Atom: FromIterator<char>,
        Self::Group: FromIterator<Self>,
    {
        parse::from_str(s)
    }

    /// Deserializes into an [`Expression`].
    #[cfg(all(feature = "parse", feature = "serde"))]
    #[cfg_attr(docsrs, doc(cfg(all(feature = "parse", feature = "serde"))))]
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
        Self: 'de,
        Self::Atom: FromIterator<char>,
        Self::Group: FromIterator<Self>,
    {
        de::deserialize(deserializer)
    }

    /// Serializes an [`Expression`].
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        Self::Atom: Serialize,
    {
        ExprRef::serialize(&self.cases(), serializer)
    }

    /// Returns the kind of an expression.
    #[must_use]
    #[inline]
    fn kind(&self) -> ExprKind {
        ExprRef::kind(&self.cases())
    }

    /// Checks if the [`Expression`] is atomic.
    #[must_use]
    #[inline]
    fn is_atom(&self) -> bool {
        ExprRef::is_atom(&self.cases())
    }

    /// Checks if the [`Expression`] is a grouped expression.
    #[must_use]
    #[inline]
    fn is_group(&self) -> bool {
        ExprRef::is_group(&self.cases())
    }

    /// Converts from an [`Expression`] to an
    /// [`Option`]`<`[`E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    fn atom(self) -> Option<Self::Atom> {
        Expr::atom(self.into())
    }

    /// Converts from an [`&Expression`](Expression) to an
    /// [`Option`]`<`[`&Expression::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    fn atom_ref(&self) -> Option<&Self::Atom> {
        ExprRef::atom(self.cases())
    }

    /// Converts from an [`Expression`] to an
    /// [`Option`]`<`[`Expression::Group`](Expression::Group)`>`.
    #[must_use]
    #[inline]
    fn group(self) -> Option<Self::Group> {
        Expr::group(self.into())
    }

    /// Converts from an [`&Expression`](Expression) to an
    /// [`Option`]`<`[`GroupRef`](GroupReference)`>`.
    #[must_use]
    #[inline]
    fn group_ref(&self) -> Option<GroupRef<Self>> {
        ExprRef::group(self.cases())
    }

    /// Returns the contained [`Atom`](Expression::Atom) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a [`Group`](Expression::Group).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    fn unwrap_atom(self) -> Self::Atom {
        Expr::unwrap_atom(self.into())
    }

    /// Returns the contained [`Group`](Expression::Group) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an [`Atom`](Expression::Atom).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    fn unwrap_group(self) -> Self::Group {
        Expr::unwrap_group(self.into())
    }

    /// Builds an empty atomic expression.
    #[inline]
    fn default_atom<T>() -> Self::Atom
    where
        Self::Atom: FromIterator<T>,
    {
        None.into_iter().collect()
    }

    /// Builds an empty grouped expression.
    #[inline]
    fn default_group() -> Self::Group
    where
        Self::Group: FromIterator<Self>,
    {
        Self::Group::empty()
    }

    /// Returns the default value of an [`Expression`]: the empty group.
    #[inline]
    fn empty() -> Self
    where
        Self::Group: FromIterator<Self>,
    {
        Self::empty_group()
    }

    /// Builds an empty atomic expression.
    #[inline]
    fn empty_atom<T>() -> Self
    where
        Self::Atom: FromIterator<T>,
    {
        Self::from_atom(Self::default_atom::<T>())
    }

    /// Builds an empty grouped expression.
    #[inline]
    fn empty_group() -> Self
    where
        Self::Group: FromIterator<Self>,
    {
        Self::from_group(Self::default_group())
    }

    /// Clones an [`Expression`] that has [`Clone`]-able [`Atoms`](Expression::Atom).
    #[inline]
    fn clone(&self) -> Self
    where
        Self::Atom: Clone,
        Self::Group: FromIterator<Self>,
    {
        ExprRef::to_owned(self.cases())
    }

    /// Checks if two [`Expressions`](Expression) are equal using [`PartialEq`]
    /// on their [`Atoms`](Expression::Atom).
    #[inline]
    fn eq<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
    {
        self.cases().eq(&other.cases())
    }

    /// Checks, in parallel, if two [`Expressions`](Expression) are equal using [`PartialEq`]
    /// on their [`Atoms`](Expression::Atom).
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    fn parallel_eq<E>(&self, other: &E) -> bool
    where
        for<'g> GroupRef<'g, Self>: IndexedParallelGroupReference<Self>,
        E: Expression,
        for<'g> GroupRef<'g, E>: IndexedParallelGroupReference<E>,
        Self::Atom: PartialEq<E::Atom>,
    {
        self.cases().parallel_eq(&other.cases())
    }

    /// Inserts an expression at the location given by `cursor`.
    #[inline]
    fn insert_at<C>(self, cursor: C, expr: Self) -> Result<Self, InsertionError>
    where
        Self::Group: FromIterator<Self> + IntoIterator<Item = Self>,
        C: IntoIterator<Item = usize>,
    {
        // TODO: add `insert_at_with` so that we only construct the `expr` if we need it
        self.into().insert_at(cursor, expr)
    }

    /// Inserts an expression at the location given by `cursor`.
    #[inline]
    fn insert_at_ref<C>(&self, cursor: C, expr: Self) -> Result<Self, InsertionError>
    where
        Self::Atom: Clone,
        Self::Group: FromIterator<Self>,
        C: IntoIterator<Item = usize>,
    {
        // TODO: add `insert_at_ref_with` so that we only construct the `expr` if we need it
        self.cases().insert_at_ref(cursor, expr)
    }

    /// Checks if an [`Expression`] is a sub-tree of another [`Expression`] using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[inline]
    fn is_subexpression<E>(&self, other: &E) -> bool
    where
        E: Expression,
        Self::Atom: PartialEq<E::Atom>,
    {
        self.cases().is_subexpression(&other.cases())
    }

    /// Checks, in parallel, if an [`Expression`] is a sub-tree of another [`Expression`] using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    fn parallel_is_subexpression<E>(&self, other: &E) -> bool
    where
        for<'g> GroupRef<'g, Self>: Sync + IndexedParallelGroupReference<Self>,
        E: Expression,
        for<'g> GroupRef<'g, E>: IndexedParallelGroupReference<E>,
        Self::Atom: Sync + PartialEq<E::Atom>,
    {
        self.cases().parallel_is_subexpression(&other.cases())
    }

    /// Checks if expression matches given [`Pattern`](pattern::Pattern).
    #[cfg(feature = "pattern")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pattern")))]
    #[inline]
    fn matches<P>(&self, pattern: P) -> bool
    where
        P: pattern::Pattern<Self>,
    {
        pattern.matches(self)
    }

    /// Checks if `self` matches an [equality pattern](pattern::EqualExpressionPattern).
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

    /// Checks if `self` matches a [subexpression pattern](pattern::SubExpressionPattern).
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

    /// Checks if `self` matches a [basic shape pattern](pattern::BasicShape).
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

    /// Checks if `self` matches a [wildcard expression](pattern::WildCardPattern).
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

    /// Extends a function on [`Atoms`](Expression::Atom) to a
    /// function on [`Expressions`](Expression).
    #[inline]
    fn map<E, F>(self, f: F) -> E
    where
        Self::Group: IntoIterator<Item = Self>,
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(Self::Atom) -> E::Atom,
    {
        Expr::map(self.into(), f)
    }

    /// Extends a function on [`&Atoms`](Expression::Atom) to a
    /// function on [`&Expressions`](Expression).
    #[inline]
    fn map_ref<E, F>(&self, f: F) -> E
    where
        E: Expression,
        E::Group: FromIterator<E>,
        F: FnMut(&Self::Atom) -> E::Atom,
    {
        ExprRef::map_ref(&self.cases(), f)
    }

    /// Substitutes an [`Expression`] into each [`Atom`](Expression::Atom) of `self`.
    #[inline]
    fn substitute<F>(self, f: F) -> Self
    where
        Self::Group: FromIterator<Self> + IntoIterator<Item = Self>,
        F: FnMut(Self::Atom) -> Self,
    {
        Expr::substitute(self.into(), f)
    }

    /// Substitutes an [`Expression`] into each [`&Atom`](Expression::Atom) of `&self`.
    #[inline]
    fn substitute_ref<F>(&self, f: F) -> Self
    where
        Self::Group: FromIterator<Self>,
        F: FnMut(&Self::Atom) -> Self,
    {
        ExprRef::substitute_ref(&self.cases(), f)
    }

    /// Substitutes, in parallel, an [`Expression`] into each [`&Atom`](Expression::Atom) of `&self`.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    fn parallel_substitute_ref<F>(&self, f: F) -> Self
    where
        Self: Send,
        Self::Group: FromParallelIterator<Self>,
        for<'r> GroupRef<'r, Self>: ParallelGroupReference<Self>,
        F: Send + Sync + Fn(&Self::Atom) -> Self,
    {
        ExprRef::parallel_substitute_ref(&self.cases(), f)
    }
}

/// Multi-Expressions Module
#[cfg(feature = "multi")]
#[cfg_attr(docsrs, doc(cfg(feature = "multi")))]
pub mod multi {
    use super::*;

    /// MultiGroup Reference
    pub trait MultiGroupReference<E>: GroupReference<E>
    where
        E: Expression,
    {
        /// MultiGroup Kind
        type Kind<'e>;

        /// Returns the multi-group kind.
        fn kind(&self) -> Self::Kind<'_>;
    }

    /// MultiGroup Trait
    pub trait MultiGroup<E>: Group<E>
    where
        E: Expression,
    {
        /// Multi Reference Type
        type MultiRef<'e>: MultiGroupReference<E>
        where
            E: 'e;

        /// Returns a inner reference to the multi group.
        fn multi_reference(&self) -> Self::MultiRef<'_>;
    }

    /// Multi-Expression Trait
    pub trait MultiExpression: Expression
    where
        <Self as Expression>::Group: MultiGroup<Self>,
    {
    }
}

/// Expression Insertion Error
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InsertionError {
    /// Error when trying to insert with an empty cursor on a grouped expression
    EmptyGrouped,

    /// Error when trying to insert with a non-empty cursor on an atomic expression
    NonEmptyAtomic,
}

impl InsertionError {
    /// Checks if the insertion would be invalid upon invocation.
    #[inline]
    pub fn check_invocation<E>(
        base: Expr<E>,
        first: impl Into<Option<usize>>,
    ) -> Result<Option<(E::Group, usize)>, Self>
    where
        E: Expression,
    {
        // FIXME: make the calling convention friendlier
        match (base.group(), first.into()) {
            (Some(group), Some(first)) => Ok(Some((group, first))),
            (Some(_), None) => Err(Self::EmptyGrouped),
            (None, Some(_)) => Err(Self::NonEmptyAtomic),
            (None, None) => Ok(None),
        }
    }

    /// Checks if the insertion would be invalid upon invocation.
    #[inline]
    pub fn check_invocation_ref<'b, 'e, E>(
        base: &'b ExprRef<'e, E>,
        first: impl Into<Option<usize>>,
    ) -> Result<Option<(&'b GroupRef<'e, E>, usize)>, Self>
    where
        E: Expression,
    {
        // FIXME: make the calling convention friendlier
        match (base.group_ref(), first.into()) {
            (Some(group), Some(first)) => Ok(Some((group, first))),
            (Some(_), None) => Err(Self::EmptyGrouped),
            (None, Some(_)) => Err(Self::NonEmptyAtomic),
            (None, None) => Ok(None),
        }
    }
}

/// Internal Reference to an [`Expression`] Type
pub enum ExprRef<'e, E>
where
    E: 'e + Expression,
{
    /// Reference to an atomic expression
    Atom(&'e E::Atom),

    /// Reference to a grouped expression
    Group(GroupRef<'e, E>),
}

impl<'e, E> ExprRef<'e, E>
where
    E: Expression,
{
    /// Returns the kind of an expression.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> ExprKind {
        self.into()
    }

    /// Checks if the [`ExprRef`] is an atomic expression [`&Atom`](Expression::Atom).
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Self::Atom(_))
    }

    /// Checks if the [`ExprRef`] is a grouped expression [`Group<E>::Ref`].
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, Self::Group(_))
    }

    /// Converts from an [`ExprRef<'e, E>`] to an
    /// [`Option`]`<`[`&'e E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<&'e E::Atom> {
        match self {
            Self::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an [`&ExprRef<'e, E>`](ExprRef) to an
    /// [`Option`]`<`[`&'e E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    pub fn atom_ref(&self) -> Option<&'e E::Atom> {
        match self {
            Self::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an [`ExprRef<'e, E>`] to an
    /// [`Option`]`<`[`GroupRef<'e, E>`]`>`.
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<GroupRef<'e, E>> {
        match self {
            Self::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Converts from an [`&ExprRef<'e, E>`] to an
    /// [`Option`]`<`[`&GroupRef<'e, E>`]>`.
    #[must_use]
    #[inline]
    pub fn group_ref(&self) -> Option<&GroupRef<'e, E>> {
        match self {
            Self::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Returns the contained [`&'e Atom`](Expression::Atom) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a [`Group`](Expression::Group).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> &'e E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained [`GroupRef<'e, E>`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an [`Atom`](Expression::Atom).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> GroupRef<'e, E> {
        self.group().unwrap()
    }

    /// Returns new owned copy of the underlying expression.
    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_owned(self) -> E
    where
        E::Atom: Clone,
        E::Group: FromIterator<E>,
    {
        E::from_expr(self.into())
    }

    /// Extends a function on [`&Atoms`](Expression::Atom) to a
    /// function on [`&Expressions`](Expression).
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
                    .map(move |e| e.cases().map_ref_inner(f))
                    .collect(),
            ),
        }
    }

    /// Substitutes an [`Expression`] into each [`&Atom`](Expression::Atom) of `&self`.
    #[inline]
    pub fn substitute_ref<F>(&self, mut f: F) -> E
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        self.substitute_ref_inner(&mut f)
    }

    #[inline]
    fn substitute_ref_group_inner<I, F>(iter: I, f: &mut F) -> E::Group
    where
        E: 'e,
        E::Group: FromIterator<E>,
        I: Iterator,
        I::Item: Reference<'e, E>,
        F: FnMut(&E::Atom) -> E,
    {
        iter.map(move |e| e.cases().substitute_ref_inner(f))
            .collect()
    }

    #[inline]
    fn substitute_ref_inner<F>(&self, f: &mut F) -> E
    where
        E::Group: FromIterator<E>,
        F: FnMut(&E::Atom) -> E,
    {
        match self {
            Self::Atom(atom) => f(atom),
            Self::Group(group) => {
                E::from_group(ExprRef::substitute_ref_group_inner(group.iter(), f))
            }
        }
    }

    /// Substitutes, in parallel, an [`Expression`] into each [`&Atom`](Expression::Atom) of `&self`.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    pub fn parallel_substitute_ref<F>(&self, f: F) -> E
    where
        E: Send,
        E::Group: FromParallelIterator<E>,
        for<'r> GroupRef<'r, E>: ParallelGroupReference<E>,
        F: Send + Sync + Fn(&E::Atom) -> E,
    {
        self.parallel_substitute_ref_inner(&f)
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    fn parallel_substitute_ref_group_inner<I, F>(iter: I, f: &F) -> E::Group
    where
        E: 'e + Send,
        E::Group: FromParallelIterator<E>,
        for<'r> GroupRef<'r, E>: ParallelGroupReference<E>,
        I: ParallelIterator,
        I::Item: Reference<'e, E>,
        F: Send + Sync + Fn(&E::Atom) -> E,
    {
        iter.map(move |e| e.cases().parallel_substitute_ref_inner(f))
            .collect()
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    fn parallel_substitute_ref_inner<F>(&self, f: &F) -> E
    where
        E: Send,
        E::Group: FromParallelIterator<E>,
        for<'r> GroupRef<'r, E>: ParallelGroupReference<E>,
        F: Send + Sync + Fn(&E::Atom) -> E,
    {
        match self {
            Self::Atom(atom) => f(atom),
            Self::Group(group) => E::from_group(ExprRef::parallel_substitute_ref_group_inner(
                group.par_iter(),
                f,
            )),
        }
    }

    /// Inserts an expression at the location given by `cursor`.
    #[inline]
    pub fn insert_at_ref<C>(&self, cursor: C, expr: E) -> Result<E, InsertionError>
    where
        E::Atom: Clone,
        E::Group: FromIterator<E>,
        C: IntoIterator<Item = usize>,
    {
        let mut cursor = cursor.into_iter();
        self.insert_at_ref_inner(cursor.next(), &mut cursor, expr)
    }

    #[inline]
    fn insert_at_ref_inner<C>(
        &self,
        next: Option<usize>,
        cursor: &mut C,
        expr: E,
    ) -> Result<E, InsertionError>
    where
        E::Atom: Clone,
        E::Group: FromIterator<E>,
        C: Iterator<Item = usize>,
    {
        // TODO: call `insert_at_ref_inner` using `E` so that we can override its behavior for
        // optimization
        match InsertionError::check_invocation_ref(self, next)? {
            Some((group, first)) => util::map_once_at_index(
                group.iter(),
                first,
                move |e| Ok(e.to_owned()),
                move |e| e.cases().insert_at_ref_inner(cursor.next(), cursor, expr),
            )
            .collect::<Result<_, _>>()
            .map(E::from_group),
            _ => Ok(expr),
        }
    }

    /// Checks if an [`Expression`] is a sub-tree of another [`Expression`] using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    pub fn is_subexpression<'r, R>(&self, other: &ExprRef<'r, R>) -> bool
    where
        R: Expression,
        E::Atom: PartialEq<R::Atom>,
    {
        match self {
            Self::Atom(atom) => match other {
                ExprRef::Atom(other) => atom == other,
                ExprRef::Group(other) => {
                    other.iter().any(move |e| self.is_subexpression(&e.cases()))
                }
            },
            Self::Group(group) => match other.group_ref() {
                Some(other) => {
                    other.iter().any(move |e| self.is_subexpression(&e.cases()))
                        || Self::eq_groups::<R>(group, other)
                }
                _ => false,
            },
        }
    }

    /// Checks, in parallel, if an [`Expression`] is a sub-tree of another [`Expression`] using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    pub fn parallel_is_subexpression<'r, R>(&self, other: &ExprRef<'r, R>) -> bool
    where
        E::Atom: Sync,
        for<'g> GroupRef<'g, E>: Sync + IndexedParallelGroupReference<E>,
        R: Expression,
        for<'g> GroupRef<'g, R>: IndexedParallelGroupReference<R>,
        E::Atom: PartialEq<R::Atom>,
    {
        match self {
            Self::Atom(atom) => match other {
                ExprRef::Atom(other) => atom == other,
                ExprRef::Group(other) => other
                    .indexed_par_iter()
                    .any(move |e| self.parallel_is_subexpression(&e.cases())),
            },
            Self::Group(group) => match other.group_ref() {
                Some(other) => {
                    other
                        .indexed_par_iter()
                        .any(move |e| self.parallel_is_subexpression(&e.cases()))
                        || Self::parallel_eq_groups::<R>(group, other)
                }
                _ => false,
            },
        }
    }

    /// Checks if two [`Groups`](Expression::Group) are equal pointwise.
    #[inline]
    pub fn eq_groups<'r, R>(lhs: &GroupRef<'e, E>, rhs: &GroupRef<'r, R>) -> bool
    where
        R: Expression,
        E::Atom: PartialEq<R::Atom>,
    {
        util::eq_by(lhs.iter(), rhs.iter(), move |l, r| l.cases().eq(&r.cases()))
    }

    /// Checks, in parallel, if two [`Expressions`](Expression) are equal using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    pub fn parallel_eq<'r, R>(&self, other: &ExprRef<'r, R>) -> bool
    where
        for<'g> GroupRef<'g, E>: IndexedParallelGroupReference<E>,
        R: Expression,
        for<'g> GroupRef<'g, R>: IndexedParallelGroupReference<R>,
        E::Atom: PartialEq<R::Atom>,
    {
        match (self, other) {
            (Self::Atom(lhs), ExprRef::Atom(rhs)) => *lhs == *rhs,
            (Self::Group(lhs), ExprRef::Group(rhs)) => Self::parallel_eq_groups::<R>(lhs, rhs),
            _ => false,
        }
    }

    /// Checks, in parallel, if two [`Groups`](Expression::Group) are equal pointwise.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    pub fn parallel_eq_groups<'r, R>(lhs: &GroupRef<'e, E>, rhs: &GroupRef<'r, R>) -> bool
    where
        for<'g> GroupRef<'g, E>: IndexedParallelGroupReference<E>,
        R: Expression,
        for<'g> GroupRef<'g, R>: IndexedParallelGroupReference<R>,
        E::Atom: PartialEq<R::Atom>,
    {
        lhs.indexed_par_iter()
            .zip(rhs.indexed_par_iter())
            .all(move |(l, r)| l.cases().parallel_eq(&r.cases()))
    }
}

impl<'e, E> Clone for ExprRef<'e, E>
where
    E: Expression,
    GroupRef<'e, E>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        match &self {
            Self::Atom(atom) => Self::Atom(atom),
            Self::Group(group) => Self::Group(group.clone()),
        }
    }
}

impl<'e, E> Copy for ExprRef<'e, E>
where
    E: Expression,
    GroupRef<'e, E>: Copy,
{
}

impl<'l, 'r, L, R> PartialEq<ExprRef<'r, R>> for ExprRef<'l, L>
where
    L: Expression,
    R: Expression,
    L::Atom: PartialEq<R::Atom>,
{
    /// Checks if two [`Expressions`](Expression) are equal using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[inline]
    fn eq(&self, other: &ExprRef<'r, R>) -> bool {
        match (self, other) {
            (Self::Atom(lhs), ExprRef::Atom(rhs)) => *lhs == *rhs,
            (Self::Group(lhs), ExprRef::Group(rhs)) => Self::eq_groups::<R>(lhs, rhs),
            _ => false,
        }
    }
}

impl<'e, E> From<&'e E> for ExprRef<'e, E>
where
    E: Expression,
{
    #[inline]
    fn from(expr: &'e E) -> Self {
        expr.cases()
    }
}

impl<'e, E> From<&'e Expr<E>> for ExprRef<'e, E>
where
    E: Expression,
{
    #[must_use]
    #[inline]
    fn from(expr: &'e Expr<E>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::Atom(atom),
            Expr::Group(group) => Self::Group(group.as_ref()),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'e, E> Serialize for ExprRef<'e, E>
where
    E: Expression,
    E::Atom: Serialize,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self {
            Self::Atom(atom) => atom.serialize(serializer),
            Self::Group(group) => serializer.collect_seq(group.iter().map(move |e| e.cases())),
        }
    }
}

/// Internal Mutable Reference to an [`Expression`] Type
pub enum ExprRefMut<'e, E>
where
    E: 'e + Expression,
    E::Group: GroupMut<E>,
{
    /// Mutable reference to an atomic expression
    Atom(&'e mut E::Atom),

    /// Mutable reference to a grouped expression
    Group(GroupRefMut<'e, E>),
}

impl<'e, E> From<&'e mut E> for ExprRefMut<'e, E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    #[inline]
    fn from(expr: &'e mut E) -> Self {
        expr.cases_mut()
    }
}

impl<'e, E> From<&'e mut Expr<E>> for ExprRefMut<'e, E>
where
    E: Expression,
    E::Group: GroupMut<E>,
{
    #[must_use]
    #[inline]
    fn from(expr: &'e mut Expr<E>) -> Self {
        match expr {
            Expr::Atom(atom) => Self::Atom(atom),
            Expr::Group(group) => Self::Group(group.as_mut()),
        }
    }
}

/// Canonical Concrete [`Expression`] Type
#[derive(Debug)]
pub enum Expr<E>
where
    E: Expression,
{
    /// Atomic element
    Atom(E::Atom),

    /// Grouped expression
    Group(E::Group),
}

impl<E> Expr<E>
where
    E: Expression,
{
    /// Returns the kind of an expression.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> ExprKind {
        self.into()
    }

    /// Checks if the [`Expr`] is an atomic expression.
    #[must_use]
    #[inline]
    pub fn is_atom(&self) -> bool {
        matches!(self, Expr::Atom(_))
    }

    /// Checks if the [`Expr`] is a grouped expression.
    #[must_use]
    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self, Expr::Group(_))
    }

    /// Converts from an [`Expr<E>`] to an [`Option`]`<`[`E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    pub fn atom(self) -> Option<E::Atom> {
        match self {
            Expr::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an [`&Expr<E>`](Expr) to an [`Option`]`<`[`&E::Atom`](Expression::Atom)`>`.
    #[must_use]
    #[inline]
    pub fn atom_ref(&self) -> Option<&E::Atom> {
        match self {
            Expr::Atom(atom) => Some(atom),
            _ => None,
        }
    }

    /// Converts from an [`Expr<E>`] to an [`Option`]`<`[`E::Group`](Expression::Group)`>`.
    #[must_use]
    #[inline]
    pub fn group(self) -> Option<E::Group> {
        match self {
            Expr::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Converts from an [`&Expr<E>`](Expr) to an [`Option`]`<`[`&E::Group`](Expression::Group)`>`.
    #[must_use]
    #[inline]
    pub fn group_ref(&self) -> Option<&E::Group> {
        match self {
            Expr::Group(group) => Some(group),
            _ => None,
        }
    }

    /// Returns the contained [`Atom`](Expression::Atom) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is a [`Group`](Expression::Group).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    pub fn unwrap_atom(self) -> E::Atom {
        self.atom().unwrap()
    }

    /// Returns the contained [`Group`](Expression::Group) value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the `self` value is an [`Atom`](Expression::Atom).
    #[cfg(feature = "panic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "panic")))]
    #[inline]
    #[track_caller]
    pub fn unwrap_group(self) -> E::Group {
        self.group().unwrap()
    }

    /// Extends a function on [`Atoms`](Expression::Atom) to a
    /// function on [`Expressions`](Expression).
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
            Self::Atom(atom) => O::from_atom(f(atom)),
            Self::Group(group) => O::from_group(
                group
                    .into_iter()
                    .map(move |e| e.into().map_inner(f))
                    .collect(),
            ),
        }
    }

    /// Substitutes an [`Expression`] into each [`Atom`](Expression::Atom) of `self`.
    #[inline]
    pub fn substitute<F>(self, mut f: F) -> E
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        self.substitute_inner(&mut f)
    }

    #[inline]
    fn substitute_group_inner<I, F>(iter: I, f: &mut F) -> E::Group
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        I: Iterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        iter.map(move |e| e.into().substitute_inner(f)).collect()
    }

    #[inline]
    fn substitute_inner<F>(self, f: &mut F) -> E
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        F: FnMut(E::Atom) -> E,
    {
        match self {
            Self::Atom(atom) => f(atom),
            Self::Group(group) => E::from_group(Self::substitute_group_inner(group.into_iter(), f)),
        }
    }

    /// Inserts an expression at the location given by `cursor`.
    #[inline]
    pub fn insert_at<C>(self, cursor: C, expr: E) -> Result<E, InsertionError>
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        C: IntoIterator<Item = usize>,
    {
        let mut cursor = cursor.into_iter();
        self.insert_at_inner(cursor.next(), &mut cursor, expr)
    }

    #[inline]
    fn insert_at_inner<C>(
        self,
        next: Option<usize>,
        cursor: &mut C,
        expr: E,
    ) -> Result<E, InsertionError>
    where
        E::Group: FromIterator<E> + IntoIterator<Item = E>,
        C: Iterator<Item = usize>,
    {
        // TODO: call `insert_at_inner` using `E` so that we can override its behavior for
        // optimization
        match InsertionError::check_invocation(self, next)? {
            Some((group, first)) => {
                util::map_once_at_index(group.into_iter(), first, Ok, move |e| {
                    e.into().insert_at_inner(cursor.next(), cursor, expr)
                })
                .collect::<Result<_, _>>()
                .map(E::from_group)
            }
            _ => Ok(expr),
        }
    }
}

impl<E> Clone for Expr<E>
where
    E: Expression,
    E::Atom: Clone,
    E::Group: FromIterator<E>,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Atom(atom) => Self::Atom(atom.clone()),
            Self::Group(group) => Self::Group(group.clone()),
        }
    }
}

impl<E> Default for Expr<E>
where
    E: Expression,
    E::Group: FromIterator<E>,
{
    /// Returns the empty group expression.
    #[inline]
    fn default() -> Self {
        E::empty().into()
    }
}

impl<L, R> PartialEq<Expr<R>> for Expr<L>
where
    L: Expression,
    R: Expression,
    L::Atom: PartialEq<R::Atom>,
{
    /// Checks if two [`Expressions`](Expression) are equal using
    /// [`PartialEq`] on their [`Atoms`](Expression::Atom).
    #[inline]
    fn eq(&self, other: &Expr<R>) -> bool {
        ExprRef::from(self).eq(&ExprRef::from(other))
    }
}

impl<E> Eq for Expr<E>
where
    E: Expression,
    E::Atom: PartialEq,
{
}

impl<'e, E> From<ExprRef<'e, E>> for Expr<E>
where
    E: Expression,
    E::Atom: Clone,
    E::Group: FromIterator<E>,
{
    #[must_use]
    #[inline]
    fn from(expr: ExprRef<'e, E>) -> Self {
        match expr {
            ExprRef::Atom(atom) => Self::Atom(atom.clone()),
            ExprRef::Group(group) => Self::Group(group.to_owned()),
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
    type Err = parse::Error<parse::FromCharactersError>;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        E::from_str(s).map(E::into)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<E> Serialize for Expr<E>
where
    E: Expression,
    E::Atom: Serialize,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ExprRef::from(self).serialize(serializer)
    }
}

#[cfg(all(feature = "parse", feature = "serde"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "parse", feature = "serde"))))]
impl<'de, E> Deserialize<'de> for Expr<E>
where
    E: 'de + Expression,
    E::Atom: FromIterator<char>,
    E::Group: FromIterator<E>,
{
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        E::deserialize(deserializer).map(E::into)
    }
}

/// [`Expression`] Kinds
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExprKind {
    /// Atomic expression kind
    Atom,

    /// Grouped expression kind
    Group,
}

impl<'e, 'r, E> From<&'r ExprRef<'e, E>> for ExprKind
where
    E: Expression,
{
    #[inline]
    fn from(expr: &'r ExprRef<'e, E>) -> Self {
        match expr {
            ExprRef::Atom(_) => Self::Atom,
            ExprRef::Group(_) => Self::Group,
        }
    }
}

impl<'e, E> From<ExprRef<'e, E>> for ExprKind
where
    E: Expression,
{
    #[inline]
    fn from(expr: ExprRef<'e, E>) -> Self {
        Self::from(&expr)
    }
}

impl<'r, E> From<&'r Expr<E>> for ExprKind
where
    E: Expression,
{
    #[inline]
    fn from(expr: &'r Expr<E>) -> Self {
        match expr {
            Expr::Atom(_) => Self::Atom,
            Expr::Group(_) => Self::Group,
        }
    }
}

impl<E> From<Expr<E>> for ExprKind
where
    E: Expression,
{
    #[inline]
    fn from(expr: Expr<E>) -> Self {
        Self::from(&expr)
    }
}

/// Utilities Module
pub mod util {
    use core::fmt;

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

    /// Takes a closure and creates an iterator which calls that closure on each element, except
    /// for one special element, the first one to match the predicate, where the iterator calls the second
    /// closure instead.
    #[inline]
    pub fn map_first_once<B, I, M, P, F>(
        iter: I,
        map: M,
        predicate: P,
        once: F,
    ) -> MapFirstOnce<I, M, P, F>
    where
        I: Iterator,
        M: FnMut(I::Item) -> B,
        P: FnMut(&I::Item) -> bool,
        F: FnOnce(I::Item) -> B,
    {
        MapFirstOnce::new(iter, map, predicate, once)
    }

    /// Takes a closure and creates an iterator which calls that closure on each element, except
    /// for one special element at the given index, where the iterator calls the second closure
    /// instead.
    #[inline]
    pub fn map_once_at_index<B, I, M, F>(
        iter: I,
        index: usize,
        mut map: M,
        once: F,
    ) -> impl Iterator<Item = B>
    where
        I: Iterator,
        M: FnMut(I::Item) -> B,
        F: FnOnce(I::Item) -> B,
    {
        // FIXME: return an iterator structure so we can optimize further chains
        map_first_once(
            iter.enumerate(),
            move |(_, n)| map(n),
            move |(i, _)| *i == index,
            move |(_, n)| once(n),
        )
    }

    /// Map First Once Iterator
    ///
    /// This iterator is an extension of the [`Map`] iterator which allows for one special value
    /// (specified by the `predicate: P`) to be modified by a `FnOnce` function instead of by an
    /// `FnMut` function.
    ///
    /// See [`map_first_once`] for more information.
    ///
    /// [`Map`]: core::iter::Map
    //
    // TODO: implement the same optimizations/impls as `core::iter::Map`
    //
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    #[derive(Clone)]
    pub struct MapFirstOnce<I, M, P, F> {
        iter: I,
        map: M,
        predicate_and_once: Option<(P, F)>,
    }

    impl<I, M, P, F> MapFirstOnce<I, M, P, F> {
        #[inline]
        fn new(iter: I, map: M, predicate: P, once: F) -> Self {
            Self {
                iter,
                map,
                predicate_and_once: Some((predicate, once)),
            }
        }
    }

    impl<I, M, P, F> fmt::Debug for MapFirstOnce<I, M, P, F>
    where
        I: fmt::Debug,
    {
        #[inline]
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MapOnce").field("iter", &self.iter).finish()
        }
    }

    impl<B, I, M, P, F> Iterator for MapFirstOnce<I, M, P, F>
    where
        I: Iterator,
        M: FnMut(I::Item) -> B,
        P: FnMut(&I::Item) -> bool,
        F: FnOnce(I::Item) -> B,
    {
        type Item = B;

        #[inline]
        fn next(&mut self) -> Option<B> {
            match self.iter.next() {
                Some(next) => match self.predicate_and_once.take() {
                    Some((mut predicate, once)) => {
                        if predicate(&next) {
                            Some(once(next))
                        } else {
                            self.predicate_and_once = Some((predicate, once));
                            Some((self.map)(next))
                        }
                    }
                    _ => Some((self.map)(next)),
                },
                _ => None,
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }
}

/// [`Expression`] Parsing Module
#[cfg(feature = "parse")]
#[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
pub mod parse {
    use {
        super::*,
        core::{
            fmt, hash,
            iter::{empty, from_fn, FromIterator, FusedIterator, Peekable},
            marker::PhantomData,
            result,
        },
    };

    /// [`Expression`] Parsing Error
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum Error<A> {
        /// Multiple [`Expressions`](Expression) at top level
        TooManyExpressions,

        /// [`Group`](Expression::Group) was not closed
        OpenGroup,

        /// [`Group`](Expression::Group) was not opened
        UnopenedGroup,

        /// Found an empty [`Group`](Expression::Group) that was not opened or closed
        BadEmptyGroup,

        /// Found leading skip symbols
        LeadingSkipSymbols,

        /// Found trailing symbols
        TrailingSymbols,

        /// [`Group`](Expression::Group) was opened when only an
        /// [`Atom`](Expression::Atom) was expected
        BadOpenGroup,

        /// [`Atom`](Expression::Atom) was started when only a
        /// [`Group`](Expression::Group) was expected
        BadStartAtom,

        /// Parsing an [`Atom`](Expression::Atom) failed with this error
        AtomParseError(A),
    }

    impl<A> From<A> for Error<A> {
        #[inline]
        fn from(err: A) -> Self {
            Self::AtomParseError(err)
        }
    }

    /// [`Expression`] Parsing Result Type
    pub type Result<T, A> = result::Result<T, Error<A>>;

    /// [`Expression`] Parsing Symbol Types
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum SymbolType {
        /// Symbols to Read
        Read,

        /// Start of a [`Group`](Expression::Group)
        GroupOpen,

        /// End of a [`Group`](Expression::Group)
        GroupClose,

        /// Symbols to Skip
        Skip,
    }

    /// [`Expression`] Completed Tokens
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum Token<A> {
        /// Group Open Token
        GroupOpen,

        /// Atom
        Atom(A),

        /// Group Close Token
        GroupClose,
    }

    impl<A> Token<A> {
        // TODO: add an `unwrap_atom` method

        /// Returns `true` if `self` matches [`Self::Atom`].
        #[inline]
        pub fn is_atom(&self) -> bool {
            matches!(self, Self::Atom(_))
        }

        /// Converts [`Token<A>`] into [`Option<A>`].
        #[inline]
        pub fn atom(self) -> Option<A> {
            match self {
                Self::Atom(atom) => Some(atom),
                _ => None,
            }
        }
    }

    /// [`Expression`] Parser Trait
    ///
    /// The parser [`Parser<T, E>`] can parse an [`Expression`] of type `E` from
    /// an iterator over `T`.
    pub trait Parser<T, E>
    where
        E: Expression,
    {
        /// [`Atom`](Expression::Atom) Parsing Error Type
        type AtomParseError;

        /// Classifies an incoming term.
        fn classify(&mut self, term: &T) -> SymbolType;

        /// Parses an [`Expression`] from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// This function consumes the iterator expecting nothing before or after the parsed
        /// [`Expression`]. To pause parsing after one [`Expression`], try
        /// [`parse_continue`](Self::parse_continue) instead.
        fn parse<I>(&mut self, iter: I) -> Result<E, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: IntoIterator<Item = T>,
        {
            let mut iter = iter.into_iter().peekable();
            parse_check(self, &mut iter).and_then(move |_| {
                let expr = self.parse_continue(&mut iter);
                iter.next()
                    .map(move |_| Err(Error::TooManyExpressions))
                    .unwrap_or(expr)
            })
        }

        /// Tries to parse an [`Expression`] from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// The iterator may still have elements remaining after parsing one [`Expression`].
        /// To parse exactly one [`Expression`], consuming the incoming iterator, try
        /// [`parse`](Self::parse) instead.
        fn parse_continue<I>(&mut self, iter: &mut Peekable<I>) -> Result<E, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: Iterator<Item = T>,
        {
            match iter.peek() {
                Some(peek) => match self.classify(peek) {
                    SymbolType::GroupClose => Err(Error::UnopenedGroup),
                    SymbolType::GroupOpen => self.parse_group_expression_continue(iter),
                    _ => self.parse_atom_expression_continue(iter),
                },
                _ => self.parse_atom_expression_continue(&mut empty().peekable()),
            }
        }

        /// Parses an [`Atom`](Expression::Atom) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// This function consumes the iterator expecting nothing before or after the parsed
        /// [`Atom`](Expression::Atom). To pause parsing after one [`Atom`](Expression::Atom), try
        /// [`parse_atom_continue`](Self::parse_atom_continue) instead.
        fn parse_atom<I>(&mut self, iter: I) -> Result<E::Atom, Self::AtomParseError>
        where
            I: IntoIterator<Item = T>,
        {
            let mut iter = iter.into_iter().peekable();
            if let Some(peek) = iter.peek() {
                match self.classify(peek) {
                    SymbolType::Skip => return Err(Error::LeadingSkipSymbols),
                    SymbolType::GroupClose => return Err(Error::UnopenedGroup),
                    SymbolType::GroupOpen => return Err(Error::BadOpenGroup),
                    _ => {}
                }
            }
            let atom = self.parse_atom_continue(&mut iter);
            iter.next()
                .map(move |_| Err(Error::TrailingSymbols))
                .unwrap_or_else(|| atom.map_err(Into::into))
        }

        /// Parses an [`Atom`](Expression::Atom) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// This function consumes the iterator expecting nothing before or after the parsed
        /// [`Atom`](Expression::Atom). To pause parsing after one [`Atom`](Expression::Atom), try
        /// [`parse_atom_continue`](Self::parse_atom_continue) instead.
        #[inline]
        fn parse_atom_expression<I>(&mut self, iter: I) -> Result<E, Self::AtomParseError>
        where
            I: IntoIterator<Item = T>,
        {
            self.parse_atom(iter).map(E::from_atom)
        }

        /// Tries to parse an [`Atom`](Expression::Atom) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// The iterator may still have elements remaining after parsing one [`Atom`](Expression::Atom).
        /// To parse exactly one [`Atom`](Expression::Atom), consuming the incoming iterator, try
        /// [`parse_atom`](Self::parse_atom) instead.
        fn parse_atom_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> result::Result<E::Atom, Self::AtomParseError>
        where
            I: Iterator<Item = T>;

        /// Tries to parse an [`Atom`](Expression::Atom) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// The iterator may still have elements remaining after parsing one [`Atom`](Expression::Atom).
        /// To parse exactly one [`Atom`](Expression::Atom), consuming the incoming iterator, try
        /// [`parse_atom`](Self::parse_atom) instead.
        #[inline]
        fn parse_atom_expression_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> Result<E, Self::AtomParseError>
        where
            I: Iterator<Item = T>,
        {
            self.parse_atom_continue(iter)
                .map(E::from_atom)
                .map_err(Into::into)
        }

        /// Parses an [`Group`](Expression::Group) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// This function consumes the iterator expecting nothing before or after the parsed
        /// [`Group`](Expression::Group). To pause parsing after one [`Group`](Expression::Group), try
        /// [`parse_group_continue`](Self::parse_group_continue) instead.
        fn parse_group<I>(&mut self, iter: I) -> Result<E::Group, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: IntoIterator<Item = T>,
        {
            let mut iter = iter.into_iter().peekable();
            let group = self.parse_group_continue(&mut iter);
            iter.next()
                .map(move |_| Err(Error::TrailingSymbols))
                .unwrap_or(group)
        }

        /// Parses an [`Group`](Expression::Group) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// This function consumes the iterator expecting nothing before or after the parsed
        /// [`Group`](Expression::Group). To pause parsing after one [`Group`](Expression::Group), try
        /// [`parse_group_continue`](Self::parse_group_continue) instead.
        #[inline]
        fn parse_group_expression<I>(&mut self, iter: I) -> Result<E, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: IntoIterator<Item = T>,
        {
            self.parse_group(iter).map(E::from_group)
        }

        /// Tries to parse a [`Group`](Expression::Group) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// The iterator may still have elements remaining after parsing one [`Group`](Expression::Group).
        /// To parse exactly one [`Group`](Expression::Group), consuming the incoming iterator, try
        /// [`parse_group`](Self::parse_group) instead.
        #[inline]
        fn parse_group_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> Result<E::Group, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: Iterator<Item = T>,
        {
            parse_group_continue_impl::<_, _, _, _, E, _>(
                self,
                iter,
                &Self::classify,
                &Self::parse_atom_expression_continue,
            )
        }

        /// Tries to parse a [`Group`](Expression::Group) from an [`Iterator`] over `T`.
        ///
        /// # Note
        ///
        /// The iterator may still have elements remaining after parsing one [`Group`](Expression::Group).
        /// To parse exactly one [`Group`](Expression::Group), consuming the incoming iterator, try
        /// [`parse_group`](Self::parse_group) instead.
        fn parse_group_expression_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> Result<E, Self::AtomParseError>
        where
            E::Group: FromIterator<E>,
            I: Iterator<Item = T>,
        {
            self.parse_group_continue(iter).map(E::from_group)
        }

        /// Parses [`Expression Tokens`](Token) from an [`Iterator`] over `T`.
        fn tokens<I>(&mut self, iter: I) -> Result<Tokens<T, E, Self, I>, Self::AtomParseError>
        where
            I: IntoIterator<Item = T>,
        {
            Tokens::new(self, iter)
        }

        /// Tries to parse [`Expression Tokens`](Token) from an [`Iterator`] over `T`.
        fn tokens_continue<'i, I>(
            &mut self,
            iter: &'i mut Peekable<I>,
        ) -> TokensContinue<'_, 'i, T, E, Self, I>
        where
            I: Iterator<Item = T>,
        {
            TokensContinue::new(self, iter)
        }
    }

    fn parse_group_continue_impl<P, I, C, PA, E, AE>(
        parser: &mut P,
        iter: &mut Peekable<I>,
        classify: &C,
        parse_atom_expression_continue: &PA,
    ) -> Result<E::Group, AE>
    where
        P: ?Sized,
        I: Iterator,
        C: Fn(&mut P, &I::Item) -> SymbolType,
        PA: Fn(&mut P, &mut Peekable<I>) -> Result<E, AE>,
        E: Expression,
        E::Group: FromIterator<E>,
    {
        match iter.peek() {
            Some(peek) => match classify(parser, peek) {
                SymbolType::Skip => Err(Error::LeadingSkipSymbols),
                SymbolType::GroupClose => Err(Error::UnopenedGroup),
                SymbolType::GroupOpen => {
                    let _ = iter.next();
                    from_fn(parse_group_continue_inner_impl(
                        parser,
                        iter,
                        classify,
                        parse_atom_expression_continue,
                    ))
                    .collect()
                }
                _ => Err(Error::BadStartAtom),
            },
            _ => Err(Error::BadEmptyGroup),
        }
    }

    #[inline]
    fn parse_group_continue_inner_impl<'i, P, I, C, PA, E, AE>(
        parser: &'i mut P,
        iter: &'i mut Peekable<I>,
        classify: &'i C,
        parse_atom_expression_continue: &'i PA,
    ) -> impl 'i + FnMut() -> Option<Result<E, AE>>
    where
        P: ?Sized,
        I: Iterator,
        C: Fn(&mut P, &I::Item) -> SymbolType,
        PA: Fn(&mut P, &mut Peekable<I>) -> Result<E, AE>,
        E: Expression,
        E::Group: FromIterator<E>,
    {
        move || loop {
            match iter.peek() {
                Some(peek) => match classify(parser, peek) {
                    SymbolType::Skip => {
                        let _ = iter.next();
                    }
                    SymbolType::GroupClose => {
                        let _ = iter.next();
                        return None;
                    }
                    SymbolType::GroupOpen => {
                        return Some(
                            parse_group_continue_impl(
                                parser,
                                iter,
                                classify,
                                parse_atom_expression_continue,
                            )
                            .map(E::from_group),
                        );
                    }
                    _ => return Some(parse_atom_expression_continue(parser, iter)),
                },
                _ => return Some(Err(Error::OpenGroup)),
            }
        }
    }

    #[inline]
    fn parse_check<T, E, P, I>(
        parser: &mut P,
        iter: &mut Peekable<I>,
    ) -> Result<(), P::AtomParseError>
    where
        E: Expression,
        P: ?Sized + Parser<T, E>,
        I: Iterator<Item = T>,
    {
        if let Some(true) = iter
            .peek()
            .map(|p| matches!(parser.classify(p), SymbolType::Skip))
        {
            Err(Error::LeadingSkipSymbols)
        } else {
            Ok(())
        }
    }

    /// [`Tokens`] Iterator Item
    pub type TokensItem<E, AtomParseError> =
        result::Result<Token<<E as Expression>::Atom>, AtomParseError>;

    #[inline]
    fn tokens_next<T, E, P, I>(
        parser: &mut P,
        iter: &mut Peekable<I>,
    ) -> Option<TokensItem<E, P::AtomParseError>>
    where
        E: Expression,
        P: Parser<T, E>,
        I: Iterator<Item = T>,
    {
        loop {
            match iter.peek() {
                Some(peek) => match parser.classify(peek) {
                    SymbolType::GroupOpen => {
                        let _ = iter.next();
                        return Some(Ok(Token::GroupOpen));
                    }
                    SymbolType::GroupClose => {
                        let _ = iter.next();
                        return Some(Ok(Token::GroupClose));
                    }
                    SymbolType::Skip => {
                        let _ = iter.next();
                        continue;
                    }
                    _ => return Some(parser.parse_atom_continue(iter).map(Token::Atom)),
                },
                _ => return None,
            }
        }
    }

    /// Token Iterator
    ///
    /// See the [`tokens`](Parser::tokens) method on [`Parser`].
    pub struct Tokens<'p, T, E, P, I>
    where
        P: ?Sized,
        I: IntoIterator,
    {
        parser: &'p mut P,
        iter: Peekable<I::IntoIter>,
        __: PhantomData<(T, E)>,
    }

    impl<'p, T, E, P, I> Tokens<'p, T, E, P, I>
    where
        E: Expression,
        P: ?Sized + Parser<T, E>,
        I: IntoIterator<Item = T>,
    {
        #[inline]
        fn new(parser: &'p mut P, iter: I) -> Result<Self, P::AtomParseError> {
            let mut iter = iter.into_iter().peekable();
            parse_check(parser, &mut iter).map(move |_| Self {
                parser,
                iter,
                __: PhantomData,
            })
        }
    }

    impl<'p, T, E, P, I> Iterator for Tokens<'p, T, E, P, I>
    where
        E: Expression,
        P: Parser<T, E>,
        I: IntoIterator<Item = T>,
    {
        type Item = TokensItem<E, P::AtomParseError>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            tokens_next(self.parser, &mut self.iter)
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }

    impl<'p, T, E, P, I> FusedIterator for Tokens<'p, T, E, P, I>
    where
        E: Expression,
        P: Parser<T, E>,
        I: IntoIterator<Item = T>,
        I::IntoIter: FusedIterator,
    {
    }

    /// Token Iterator
    ///
    /// See the [`tokens_continue`](Parser::tokens_continue) method on [`Parser`].
    pub struct TokensContinue<'p, 'i, T, E, P, I>
    where
        P: ?Sized,
        I: Iterator,
    {
        parser: &'p mut P,
        iter: &'i mut Peekable<I>,
        __: PhantomData<(T, E)>,
    }

    impl<'p, 'i, T, E, P, I> TokensContinue<'p, 'i, T, E, P, I>
    where
        P: ?Sized,
        I: Iterator,
    {
        #[inline]
        fn new(parser: &'p mut P, iter: &'i mut Peekable<I>) -> Self {
            Self {
                parser,
                iter,
                __: PhantomData,
            }
        }
    }

    impl<'p, 'i, T, E, P, I> Iterator for TokensContinue<'p, 'i, T, E, P, I>
    where
        E: Expression,
        P: Parser<T, E>,
        I: Iterator<Item = T>,
    {
        type Item = TokensItem<E, P::AtomParseError>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            tokens_next(self.parser, self.iter)
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }

    impl<'p, 'i, T, E, P, I> FusedIterator for TokensContinue<'p, 'i, T, E, P, I>
    where
        E: Expression,
        P: Parser<T, E>,
        I: FusedIterator<Item = T>,
    {
    }

    /// [`Expression`] Parser from [`Iterator`]s over [`Token`]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct FromTokens<E>(PhantomData<E>)
    where
        E: Expression;

    impl<E> FromTokens<E>
    where
        E: Expression,
    {
        /// Builds a new [`FromTokens`] parser.
        #[inline]
        pub fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<E> Clone for FromTokens<E>
    where
        E: Expression,
    {
        #[inline]
        fn clone(&self) -> Self {
            Self::new()
        }
    }

    impl<E> Copy for FromTokens<E> where E: Expression {}

    impl<E> fmt::Debug for FromTokens<E>
    where
        E: Expression,
    {
        #[inline]
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_tuple("FromTokens")
                .field(&format_args!("_"))
                .finish()
        }
    }

    impl<E> Default for FromTokens<E>
    where
        E: Expression,
    {
        #[inline]
        fn default() -> Self {
            Self::new()
        }
    }

    impl<E> Eq for FromTokens<E> where E: Expression {}

    impl<E> hash::Hash for FromTokens<E>
    where
        E: Expression,
    {
        #[inline]
        fn hash<H: hash::Hasher>(&self, state: &mut H) {
            self.0.hash(state)
        }
    }

    impl<E> PartialEq for FromTokens<E>
    where
        E: Expression,
    {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl<E> Parser<Token<E::Atom>, E> for FromTokens<E>
    where
        E: Expression,
    {
        type AtomParseError = ();

        #[inline]
        fn classify(&mut self, term: &Token<E::Atom>) -> SymbolType {
            match term {
                Token::GroupOpen => SymbolType::GroupOpen,
                Token::GroupClose => SymbolType::GroupClose,
                _ => SymbolType::Read,
            }
        }

        #[inline]
        fn parse_atom_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> result::Result<E::Atom, Self::AtomParseError>
        where
            I: Iterator<Item = Token<E::Atom>>,
        {
            match iter.peek() {
                Some(Token::Atom(_)) => Ok(iter.next().unwrap().atom().unwrap()),
                _ => Err(()),
            }
        }
    }

    /// Parses an [`Expression`] from an iterator over tokens.
    #[inline]
    pub fn from_tokens<I, E>(iter: I) -> Result<E, ()>
    where
        I: IntoIterator<Item = Token<E::Atom>>,
        E: Expression,
        E::Group: FromIterator<E>,
    {
        FromTokens::default().parse(iter)
    }

    /// Parses an expression [`Group`](Expression::Group) from an iterator over tokens.
    #[inline]
    pub fn from_tokens_as_group<I, E>(iter: I) -> Result<E::Group, ()>
    where
        I: IntoIterator<Item = Token<E::Atom>>,
        E: Expression,
        E::Group: FromIterator<E>,
    {
        FromTokens::<E>::default().parse_group(
            Some(Token::GroupOpen)
                .into_iter()
                .chain(iter)
                .chain(Some(Token::GroupClose)),
        )
    }

    /// [`Expression`] Parser from [`Iterator`]s over [`char`]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct FromCharacters {
        /// Quoting Symbol
        pub quote: char,

        /// Opening Group Symbol
        pub open: char,

        /// Closing Group Symbol
        pub close: char,
    }

    impl Default for FromCharacters {
        #[inline]
        fn default() -> Self {
            Self::new('\"', '(', ')')
        }
    }

    impl FromCharacters {
        /// Builds a new [`FromCharacters`] parser using the [`quote`](Self::quote) as a quoting
        /// symbol, and [`open`](Self::open) and [`close`](Self::close) characters as opening and
        /// closing group indicators.
        #[inline]
        pub fn new(quote: char, open: char, close: char) -> Self {
            Self { quote, open, close }
        }

        #[inline]
        fn is_quote(&self, c: &char) -> bool {
            *c == self.quote
        }

        #[inline]
        fn classify_char(&self, c: &char) -> SymbolType {
            if c.is_whitespace() {
                SymbolType::Skip
            } else if *c == self.open {
                SymbolType::GroupOpen
            } else if *c == self.close {
                SymbolType::GroupClose
            } else {
                SymbolType::Read
            }
        }

        #[inline]
        fn parse_atom_continue_inner<'i, I>(
            &'i self,
            iter: &'i mut Peekable<I>,
            inside_quote: &'i mut bool,
        ) -> impl 'i + FnMut() -> Option<char>
        where
            I: Iterator<Item = char>,
        {
            move || match iter.peek() {
                Some(peek) => {
                    if *inside_quote {
                        if self.is_quote(peek) {
                            *inside_quote = false;
                        }
                    } else {
                        match self.classify_char(peek) {
                            SymbolType::Read => {
                                if self.is_quote(peek) {
                                    *inside_quote = true;
                                }
                            }
                            _ => return None,
                        }
                    }
                    iter.next()
                }
                _ => None,
            }
        }
    }

    /// [`FromCharacters`] Parsing Error
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum FromCharactersError {
        /// Closing quote is missing
        MissingClosingQuote,
    }

    impl<E> Parser<char, E> for FromCharacters
    where
        E: Expression,
        E::Atom: FromIterator<char>,
    {
        type AtomParseError = FromCharactersError;

        #[inline]
        fn classify(&mut self, term: &char) -> SymbolType {
            self.classify_char(term)
        }

        #[inline]
        fn parse_atom_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> result::Result<E::Atom, Self::AtomParseError>
        where
            I: Iterator<Item = char>,
        {
            let mut inside_quote = false;
            let atom = from_fn(self.parse_atom_continue_inner(iter, &mut inside_quote)).collect();
            if inside_quote {
                Err(Self::AtomParseError::MissingClosingQuote)
            } else {
                Ok(atom)
            }
        }
    }

    /// Parses a string-like [`Expression`] from an iterator over characters.
    #[inline]
    pub fn from_chars<I, E>(iter: I) -> Result<E, FromCharactersError>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        FromCharacters::default().parse(iter)
    }

    /// Parses a string-like expression [`Group`](Expression::Group) from an iterator over characters.
    #[inline]
    pub fn from_chars_as_group<I, E>(iter: I) -> Result<E::Group, FromCharactersError>
    where
        I: IntoIterator<Item = char>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        let mut parser = FromCharacters::default();
        let iter = Some(parser.open)
            .into_iter()
            .chain(iter)
            .chain(Some(parser.close));
        Parser::<_, E>::parse_group(&mut parser, iter)
    }

    /// Parses a string-like [`Expression`] from a string.
    #[inline]
    pub fn from_str<S, E>(s: S) -> Result<E, FromCharactersError>
    where
        S: AsRef<str>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        from_chars(s.as_ref().chars())
    }

    /// Parses a string-like expression [`Group`](Expression::Group) from a string.
    #[inline]
    pub fn from_str_as_group<S, E>(s: S) -> Result<E::Group, FromCharactersError>
    where
        S: AsRef<str>,
        E: Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        // TODO: should we interface with `FromStr` for atoms or `FromIterator<char>`?
        from_chars_as_group::<_, E>(s.as_ref().chars())
    }

    /// [`Expression`] Parser from [`Iterator`]s over Strings
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct FromStrings<'s> {
        /// Quoting Symbol
        pub quote: &'s str,

        /// Opening Group Symbol
        pub open: &'s str,

        /// Closing Group Symbol
        pub close: &'s str,
    }

    impl Default for FromStrings<'_> {
        #[inline]
        fn default() -> Self {
            Self::new("\"", "(", ")")
        }
    }

    impl<'s> FromStrings<'s> {
        /// Builds a new [`FromStrings`] parser using the [`quote`](Self::quote) as a quoting
        /// symbol, and [`open`](Self::open) and [`close`](Self::close) strings as opening and
        /// closing group indicators.
        #[inline]
        pub fn new(quote: &'s str, open: &'s str, close: &'s str) -> Self {
            Self { quote, open, close }
        }

        #[inline]
        fn is_quote(&self, s: &str) -> bool {
            s == self.quote
        }

        #[inline]
        fn classify_string(&self, s: &str) -> SymbolType {
            if s.trim().is_empty() {
                SymbolType::Skip
            } else if s == self.open {
                SymbolType::GroupOpen
            } else if s == self.close {
                SymbolType::GroupClose
            } else {
                SymbolType::Read
            }
        }

        #[inline]
        fn parse_atom_continue_inner<'i, I>(
            &'i self,
            iter: &'i mut Peekable<I>,
            inside_quote: &'i mut bool,
        ) -> impl 'i + FnMut() -> Option<&'s str>
        where
            I: Iterator<Item = &'s str>,
        {
            move || match iter.peek() {
                Some(peek) => {
                    if *inside_quote {
                        if self.is_quote(peek) {
                            *inside_quote = false;
                        }
                    } else {
                        match self.classify_string(peek) {
                            SymbolType::Read => {
                                if self.is_quote(peek) {
                                    *inside_quote = true;
                                }
                            }
                            _ => return None,
                        }
                    }
                    iter.next()
                }
                _ => None,
            }
        }
    }

    /// [`FromStrings`] Parsing Error
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum FromStringsError {
        /// Closing quote is missing
        MissingClosingQuote,
    }

    impl<'s, E> Parser<&'s str, E> for FromStrings<'s>
    where
        E: Expression,
        E::Atom: FromIterator<&'s str>,
    {
        type AtomParseError = FromCharactersError;

        #[inline]
        fn classify(&mut self, term: &&'s str) -> SymbolType {
            self.classify_string(term)
        }

        #[inline]
        fn parse_atom_continue<I>(
            &mut self,
            iter: &mut Peekable<I>,
        ) -> result::Result<E::Atom, Self::AtomParseError>
        where
            I: Iterator<Item = &'s str>,
        {
            let mut inside_quote = false;
            let atom = from_fn(self.parse_atom_continue_inner(iter, &mut inside_quote)).collect();
            if inside_quote {
                Err(Self::AtomParseError::MissingClosingQuote)
            } else {
                Ok(atom)
            }
        }
    }

    /// Parses a string-like [`Expression`] from an iterator over strings.
    #[inline]
    pub fn from_strings<'s, I, E>(iter: I) -> Result<E, FromCharactersError>
    where
        I: IntoIterator<Item = &'s str>,
        E: Expression,
        E::Atom: FromIterator<&'s str>,
        E::Group: FromIterator<E>,
    {
        FromStrings::default().parse(iter)
    }

    /// Parses a string-like expression [`Group`](Expression::Group) from an iterator over graphemes.
    #[inline]
    pub fn from_strings_as_group<'s, I, E>(iter: I) -> Result<E::Group, FromCharactersError>
    where
        I: IntoIterator<Item = &'s str>,
        E: Expression,
        E::Atom: FromIterator<&'s str>,
        E::Group: FromIterator<E>,
    {
        let mut parser = FromStrings::default();
        let iter = Some(parser.open)
            .into_iter()
            .chain(iter)
            .chain(Some(parser.close));
        Parser::<_, E>::parse_group(&mut parser, iter)
    }

    /// Parses a string-like [`Expression`] from a string using graphemes.
    #[cfg(feature = "unicode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "unicode")))]
    #[inline]
    pub fn from_graphemes<'s, E>(s: &'s str) -> Result<E, FromCharactersError>
    where
        E: Expression,
        E::Atom: FromIterator<&'s str>,
        E::Group: FromIterator<E>,
    {
        from_strings(unicode_segmentation::UnicodeSegmentation::graphemes(
            s, true,
        ))
    }

    /// Parses a string-like expression [`Group`](Expression::Group) from a string.
    #[cfg(feature = "unicode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "unicode")))]
    #[inline]
    pub fn from_graphemes_as_group<'s, E>(s: &'s str) -> Result<E::Group, FromCharactersError>
    where
        E: Expression,
        E::Atom: FromIterator<&'s str>,
        E::Group: FromIterator<E>,
    {
        from_strings_as_group::<_, E>(unicode_segmentation::UnicodeSegmentation::graphemes(
            s, true,
        ))
    }
}

/// Serde Deserialization Module
#[cfg(all(feature = "parse", feature = "serde"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "parse", feature = "serde"))))]
pub mod de {
    use {
        super::*,
        core::{
            cmp::Ordering,
            fmt,
            hash::{Hash, Hasher},
            marker::PhantomData,
        },
        parse::{FromCharacters, Parser},
        serde::de::{self, SeqAccess},
    };

    /// Sequence Access Iterator
    #[derive(Debug)]
    pub struct SeqAccessIterator<'de, S, T>(S, PhantomData<&'de T>)
    where
        S: SeqAccess<'de>,
        T: Deserialize<'de>;

    impl<'de, S, T> SeqAccessIterator<'de, S, T>
    where
        S: SeqAccess<'de>,
        T: Deserialize<'de>,
    {
        /// Builds a new iterator over a [`SeqAccess`] object.
        #[inline]
        pub fn new(seq: S) -> Self {
            Self(seq, PhantomData)
        }
    }

    impl<'de, S, T> Iterator for SeqAccessIterator<'de, S, T>
    where
        S: SeqAccess<'de>,
        T: Deserialize<'de>,
    {
        type Item = Result<T, S::Error>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self.0.next_element() {
                Ok(item) => item.map(Ok),
                Err(err) => Some(Err(err)),
            }
        }
    }

    /// Expression Visitor
    #[derive(Debug)]
    pub struct Visitor<E>(PhantomData<E>)
    where
        E: Expression;

    impl<E> Clone for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn clone(&self) -> Self {
            Self(self.0)
        }
    }

    impl<E> Copy for Visitor<E> where E: Expression {}

    impl<E> Default for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn default() -> Self {
            Self(Default::default())
        }
    }

    impl<E> Eq for Visitor<E> where E: Expression {}

    impl<E> Hash for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.hash(state)
        }
    }

    impl<E> Ord for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.cmp(&other.0)
        }
    }

    impl<E> PartialEq for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl<E> PartialOrd for Visitor<E>
    where
        E: Expression,
    {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl<'de, E> de::Visitor<'de> for Visitor<E>
    where
        E: 'de + Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        type Value = Expr<E>;

        #[inline]
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "an expression")
        }

        fn visit_str<Err>(self, value: &str) -> Result<Self::Value, Err>
        where
            Err: de::Error,
        {
            Parser::<_, E>::parse_atom(&mut FromCharacters::default(), value.chars())
                .map(Expr::Atom)
                .map_err(move |err| {
                    Err::custom(format_args!(
                        "Unable to parse atom. Failed with error: {:?}",
                        err
                    ))
                })
        }

        fn visit_seq<S>(self, seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            SeqAccessIterator::new(seq)
                .map(move |e| e.map(E::from_expr))
                .collect::<Result<E::Group, _>>()
                .map(Expr::Group)
        }
    }

    /// Deserializes into an [`Expression`].
    #[inline]
    pub fn deserialize<'de, D, E>(deserializer: D) -> Result<E, D::Error>
    where
        D: Deserializer<'de>,
        E: 'de + Expression,
        E::Atom: FromIterator<char>,
        E::Group: FromIterator<E>,
    {
        deserializer
            .deserialize_any(Visitor::default())
            .map(E::from_expr)
    }
}

/// Shape Module
#[cfg(feature = "shape")]
#[cfg_attr(docsrs, doc(cfg(feature = "shape")))]
pub mod shape {
    use {
        super::*,
        core::convert::{TryFrom, TryInto},
    };

    // TODO: how to resolve this with the `Pattern` trait?

    /// Matcher Trait
    pub trait Matcher<E>
    where
        E: Expression,
    {
        /// Match Error Type
        type Error;

        /// Checks if the given atom matches the shape.
        fn matches_atom(atom: &E::Atom) -> Result<(), Self::Error>;

        /// Checks if the given group matches the shape.
        fn matches_group(group: GroupRef<E>) -> Result<(), Self::Error>;

        /// Checks if the given expression matches the shape.
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
    /// ```ignore
    /// matches(&expr).err() == expr.try_into().err()
    /// ```
    ///
    /// but can be weakend to the following,
    ///
    /// ```ignore
    /// matches(&expr).is_err() == expr.try_into().is_err()
    /// ```
    ///
    /// if it is impossible or inefficient to implement the stronger contract.
    pub trait Shape<E>:
        Matcher<E> + Into<Expr<E>> + TryFrom<Expr<E>, Error = <Self as Matcher<E>>::Error>
    where
        E: Expression,
    {
        /// Parses an [`Atom`](Expression::Atom) into [`Self`].
        #[inline]
        fn parse_atom(atom: E::Atom) -> Result<Self, <Self as Matcher<E>>::Error> {
            Expr::Atom(atom).try_into()
        }

        /// Parses a [`Group`](Expression::Group) into [`Self`].
        #[inline]
        fn parse_group(group: E::Group) -> Result<Self, <Self as Matcher<E>>::Error> {
            Expr::Group(group).try_into()
        }

        /// Parses an [`Expression`] into [`Self`].
        #[inline]
        fn parse_expr(expr: E) -> Result<Self, <Self as Matcher<E>>::Error> {
            expr.into().try_into()
        }
    }

    /* TODO: should we have this?
    impl<E, S> Shape<E> for S
    where
        E: Expression,
        S: Matcher<E> + Into<Expr<E>> + TryFrom<Expr<E>, Error = <Self as Matcher<E>>::Error>,
    {
    }
    */
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
        fn matches_group(&self, group: GroupRef<E>) -> bool;

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
        fn matches_group(&mut self, group: GroupRef<E>) -> bool;

        /// Checks if the pattern matches an expression.
        #[inline]
        fn matches(&mut self, expr: &E) -> bool {
            match expr.cases() {
                ExprRef::Atom(atom) => self.matches_atom(atom),
                ExprRef::Group(group) => self.matches_group(group),
            }
        }
    }

    /// Equal [`Expression`] Pattern
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct EqualExpressionPattern<'p, P>(&'p P)
    where
        P: Expression;

    impl<'p, P> EqualExpressionPattern<'p, P>
    where
        P: Expression,
    {
        #[inline]
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
        #[inline]
        fn matches_atom(&self, atom: &E::Atom) -> bool {
            self.0.atom().map_or(false, |a| a == atom)
        }

        #[inline]
        fn matches_group(&self, group: GroupRef<E>) -> bool {
            self.0
                .group()
                .map_or(false, move |g| ExprRef::<P>::eq_groups::<E>(&g, &group))
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
        #[inline]
        pub(crate) fn new(pattern: &'p P) -> Self {
            Self(pattern)
        }

        #[inline]
        fn matches_atom<E>(pattern: &ExprRef<'_, P>, atom: &E::Atom) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.atom_ref() {
                Some(pattern_atom) => pattern_atom == atom,
                _ => false,
            }
        }

        fn matches_group<E>(pattern: &ExprRef<P>, group: GroupRef<E>) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.group_ref() {
                Some(pattern_group) => {
                    group.iter().any(move |e| Self::matches(pattern, e.cases()))
                        || ExprRef::<P>::eq_groups::<E>(pattern_group, &group)
                }
                _ => group.iter().any(move |e| Self::matches(pattern, e.cases())),
            }
        }

        #[inline]
        fn matches<E>(pattern: &ExprRef<'_, P>, expr: ExprRef<'_, E>) -> bool
        where
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match expr {
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
            Self::matches_atom::<E>(&self.0.cases(), atom)
        }

        #[inline]
        fn matches_group(&self, group: GroupRef<E>) -> bool {
            Self::matches_group::<E>(&self.0.cases(), group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(&self.0.cases(), expr.cases())
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
        #[inline]
        pub(crate) fn new(is_wildcard: W, pattern: &'p P) -> Self {
            Self(is_wildcard, pattern)
        }

        #[inline]
        fn matches_atom<F, E>(is_wildcard: F, pattern: &ExprRef<'_, P>, atom: &E::Atom) -> bool
        where
            F: FnOnce(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern.atom_ref() {
                Some(pattern_atom) => is_wildcard(pattern_atom) || pattern_atom == atom,
                _ => false,
            }
        }

        fn matches_group<F, E>(mut is_wildcard: F, pattern: &ExprRef<P>, group: GroupRef<E>) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern {
                ExprRef::Atom(pattern_atom) if is_wildcard(pattern_atom) => true,
                ExprRef::Atom(_) => group
                    .iter()
                    .any(move |e| Self::matches(&mut is_wildcard, pattern, e.cases())),
                ExprRef::Group(pattern_group) => {
                    group
                        .iter()
                        .any(|e| Self::matches(&mut is_wildcard, pattern, e.cases()))
                        || util::eq_by(pattern_group.iter(), group.iter(), |p, e| {
                            Self::wildcard_equality(&mut is_wildcard, &p.cases(), &e.cases())
                        })
                }
            }
        }

        #[inline]
        fn matches<F, E>(is_wildcard: F, pattern: &ExprRef<'_, P>, expr: ExprRef<'_, E>) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match expr {
                ExprRef::Atom(atom) => Self::matches_atom::<_, E>(is_wildcard, pattern, atom),
                ExprRef::Group(group) => Self::matches_group::<_, E>(is_wildcard, pattern, group),
            }
        }

        fn wildcard_equality<F, E>(
            is_wildcard: &mut F,
            pattern: &ExprRef<'_, P>,
            expr: &ExprRef<'_, E>,
        ) -> bool
        where
            F: FnMut(&P::Atom) -> bool,
            E: Expression,
            P::Atom: PartialEq<E::Atom>,
        {
            match pattern {
                ExprRef::Atom(pattern_atom) => {
                    is_wildcard(pattern_atom)
                        || expr.atom_ref().map_or(false, move |a| *pattern_atom == a)
                }
                ExprRef::Group(pattern_group) => match expr.group_ref() {
                    Some(group) => util::eq_by(pattern_group.iter(), group.iter(), move |p, e| {
                        Self::wildcard_equality(is_wildcard, &p.cases(), &e.cases())
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
            Self::matches_atom::<_, E>(&self.0, &self.1.cases(), atom)
        }

        #[inline]
        fn matches_group(&self, group: GroupRef<E>) -> bool {
            Self::matches_group::<_, E>(&self.0, &self.1.cases(), group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(&self.0, &self.1.cases(), expr.cases())
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
            Self::matches_atom::<_, E>(&mut self.0, &self.1.cases(), atom)
        }

        #[inline]
        fn matches_group(&mut self, group: GroupRef<E>) -> bool {
            Self::matches_group::<_, E>(&mut self.0, &self.1.cases(), group)
        }

        #[inline]
        fn matches(&mut self, expr: &E) -> bool {
            Self::matches(&mut self.0, &self.1.cases(), expr.cases())
        }
    }

    /// Basic Expression Shape
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
            matches!(self, Self::Expr | Self::Atom)
        }

        /// Checks if the shape would match a group.
        #[inline]
        pub fn matches_group(&self) -> bool {
            matches!(self, Self::Expr | Self::Group)
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
        #[inline]
        pub(crate) fn new(pattern: &'p P) -> Self {
            Self(pattern)
        }

        #[inline]
        fn matches_atom<E>(pattern: ExprRef<'_, P>, atom: &E::Atom) -> bool
        where
            E: Expression,
        {
            let _ = atom;
            pattern.atom().map_or(false, BasicShape::matches_atom)
        }

        fn matches_group<E>(pattern: ExprRef<P>, group: GroupRef<E>) -> bool
        where
            E: Expression,
        {
            match pattern.cases() {
                ExprRef::Atom(pattern_atom) => pattern_atom.matches_group(),
                ExprRef::Group(pattern_group) => {
                    util::eq_by(pattern_group.iter(), group.iter(), move |p, e| {
                        Self::matches(p.cases(), e.cases())
                    })
                }
            }
        }

        #[inline]
        fn matches<E>(pattern: ExprRef<'_, P>, expr: ExprRef<'_, E>) -> bool
        where
            E: Expression,
        {
            match expr {
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
            Self::matches_atom::<E>(self.0.cases(), atom)
        }

        #[inline]
        fn matches_group(&self, group: GroupRef<E>) -> bool {
            Self::matches_group::<E>(self.0.cases(), group)
        }

        #[inline]
        fn matches(&self, expr: &E) -> bool {
            Self::matches(self.0.cases(), expr.cases())
        }
    }
}

/// [`Expression`] Graph Algorithms
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub mod graph {}

/// [`Expression`] Traversal Algorithms
#[cfg(feature = "visit")]
#[cfg_attr(docsrs, doc(cfg(feature = "visit")))]
pub mod visit {}

/// Vector [`Expressions`](Expression)
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod vec {
    use {
        super::*,
        alloc::{string::String, vec::Vec},
    };

    /// Vector [`Expression`] Type over [`Strings`](String)
    pub type StringExpr = Expr<String>;

    /// Vector [`Expression`] Type
    #[derive(Debug, Hash)]
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

    impl<A> From<Expr<A>> for crate::Expr<Expr<A>> {
        #[inline]
        fn from(expr: Expr<A>) -> Self {
            match expr {
                Expr::Atom(atom) => Self::Atom(atom),
                Expr::Group(group) => Self::Group(group),
            }
        }
    }

    impl<A> Clone for Expr<A>
    where
        A: Clone,
    {
        #[inline]
        fn clone(&self) -> Self {
            Expression::clone(self)
        }
    }

    impl<A> Default for Expr<A> {
        #[inline]
        fn default() -> Self {
            Expression::empty()
        }
    }

    impl<A> Eq for Expr<A> where A: Eq {}

    impl<A, Rhs> PartialEq<Expr<Rhs>> for Expr<A>
    where
        A: PartialEq<Rhs>,
    {
        #[inline]
        fn eq(&self, other: &Expr<Rhs>) -> bool {
            Expression::eq(self, other)
        }
    }

    #[cfg(feature = "parse")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parse")))]
    impl<A> FromStr for Expr<A>
    where
        A: FromIterator<char>,
    {
        type Err = parse::Error<parse::FromCharactersError>;

        #[inline]
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Expression::from_str(s)
        }
    }

    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl<A> Serialize for Expr<A>
    where
        A: Serialize,
    {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Expression::serialize(self, serializer)
        }
    }

    #[cfg(all(feature = "parse", feature = "serde"))]
    #[cfg_attr(docsrs, doc(cfg(all(feature = "parse", feature = "serde"))))]
    impl<'de, A> Deserialize<'de> for Expr<A>
    where
        A: 'de + FromIterator<char>,
    {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            Expression::deserialize(deserializer)
        }
    }

    /// Vec Multi-Expressions
    #[cfg(feature = "multi")]
    #[cfg_attr(docsrs, doc(cfg(feature = "multi")))]
    pub mod multi {
        use super::*;

        // TODO: implement all the derive traits on `MultiExpr` correctly, like `vec::Expr`

        /// Vector [`MultiExpression`](crate::multi::MultiExpression) over [`Strings`](String)
        pub type StringMultiExpr<G = ()> = MultiExpr<String, G>;

        /// Vector [`MultiExpression`](crate::multi::MultiExpression) Type
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

        impl<A, G> From<MultiExpr<A, G>> for crate::Expr<MultiExpr<A, G>> {
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
                Self::empty()
            }
        }

        impl<A, G> GroupReference<MultiExpr<A, G>> for (&Vec<MultiExpr<A, G>>, &G) {
            type Item<'e>
            where
                A: 'e,
                G: 'e,
            = &'e MultiExpr<A, G>;

            type Iter<'e>
            where
                A: 'e,
                G: 'e,
            = slice::Iter<'e, MultiExpr<A, G>>;

            fn iter(&self) -> Self::Iter<'_> {
                todo!()
            }
        }

        impl<A, G> Group<MultiExpr<A, G>> for (Vec<MultiExpr<A, G>>, G) {
            type Ref<'e>
            where
                Self: 'e,
            = (&'e Vec<MultiExpr<A, G>>, &'e G);

            fn as_ref(&self) -> Self::Ref<'_> {
                (&self.0, &self.1)
            }
        }
    }
}

/// Buffered [`Expressions`](Expression)
#[cfg(feature = "buffered")]
#[cfg_attr(docsrs, doc(cfg(feature = "buffered")))]
pub mod buffered {

    // TODO: implement a buffered expression for a string backing buffer: it should be the optimal
    //       expression for going to and from strings for CLIs etc.

    use {super::*, alloc::vec::Vec};

    /// Buffered [`Expression`] Type
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct Expr<T, LengthIndex = usize, ShapeIndex = usize> {
        atoms: Vec<T>,
        lengths: Vec<LengthIndex>,
        shape: Vec<ShapeIndex>,
    }

    /// TODO: implement
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExprGroup<T> {
        inner: Expr<T>,
    }

    /// TODO: implement
    pub struct ExprGroupReference<'e, T> {
        __: core::marker::PhantomData<&'e T>,
    }

    /// TODO: implement
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExprView<'t, T> {
        base: &'t Expr<T>,
        index: usize,
    }

    /// TODO: implement
    pub struct ExprViewIterator<'t, T> {
        _base: &'t Expr<T>,
        index: usize,
    }

    impl<'t, T> ExprViewIterator<'t, T> {
        fn _new(base: &'t Expr<T>) -> Self {
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

        fn size_hint(&self) -> (usize, Option<usize>) {
            todo!()
        }
    }

    impl<T> ExactSizeIterator for ExprViewIterator<'_, T> {}

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

    impl<'e, T> From<&'e Expr<T>> for ExprView<'e, T> {
        fn from(expr: &'e Expr<T>) -> Self {
            let _ = expr;
            todo!()
        }
    }

    impl<'e, T> Reference<'e, Expr<T>> for ExprView<'e, T> {
        fn cases(self) -> ExprRef<'e, Expr<T>> {
            todo!()
        }
    }

    impl<T> GroupReference<Expr<T>> for ExprGroupReference<'_, T> {
        type Item<'e>
        where
            T: 'e,
        = ExprView<'e, T>;

        type Iter<'e>
        where
            T: 'e,
        = ExprViewIterator<'e, T>;

        fn iter(&self) -> Self::Iter<'_> {
            todo!()
        }
    }

    impl<T> Group<Expr<T>> for ExprGroup<T> {
        type Ref<'e>
        where
            T: 'e,
        = ExprGroupReference<'e, T>;

        fn as_ref(&self) -> Self::Ref<'_> {
            todo!()
        }
    }

    impl<T> From<Expr<T>> for crate::Expr<Expr<T>> {
        #[inline]
        fn from(expr: Expr<T>) -> Self {
            let _ = expr;
            todo!()
        }
    }
}
