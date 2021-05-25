{- *********************************************************************
*                                                                      *
                   Size-indexed sets (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs, TypeOperators,
             TypeApplications, ScopedTypeVariables, PatternSynonyms,
             StandaloneDeriving, DeriveFoldable #-}

module SizedSet.Safe
  ( SizedSet
  , fromVec
  , member
  , isEmptySizedSet
  , findAndDelete, FindAndDeleteResult, pattern FADR_Yes, pattern FADR_No
  , size
  ) where

import Prelim
import Vec.Safe
import SNat.Safe

import Data.Type.Equality
import qualified Data.Set as Set
import Data.Coerce
import qualified Data.Foldable as Foldable

-- Invariant: no duplicates
type SizedSet :: Nat -> Ty -> Ty
newtype SizedSet n a where
  MkSizedSet :: Vec n a -> SizedSet n a

deriving instance Foldable (SizedSet n)
deriving instance Show a => Show (SizedSet n a)

-- fails when there are duplicates
fromVec :: Ord a => Vec n a -> Maybe (SizedSet n a)
fromVec v
  | has_duplicates = Nothing
  | otherwise      = Just $ MkSizedSet v
  where
    has_duplicates = length (foldMap Set.singleton v) /= length v

member :: Ord a => a -> SizedSet n a -> Bool
member x (MkSizedSet v) = elem x v

isEmptySizedSet :: SizedSet n a -> Maybe (n :~: Zero)
isEmptySizedSet (MkSizedSet Nil) = Just Refl
isEmptySizedSet _                = Nothing

type GenFindAndDeleteResult :: (Nat -> Ty -> Ty) -> Nat -> Ty -> Ty
data GenFindAndDeleteResult s n a where
  FADR_Yes :: s n a -> GenFindAndDeleteResult s (Succ n) a
  FADR_No  :: GenFindAndDeleteResult s n a

type FindAndDeleteResult = GenFindAndDeleteResult SizedSet

findAndDelete :: forall n a. Ord a => a -> SizedSet n a -> FindAndDeleteResult n a
findAndDelete needle (MkSizedSet v) = coerce (go v)
  where
    go :: Vec m a -> GenFindAndDeleteResult Vec m a
    go Nil = FADR_No
    go (x :> xs)
      | needle == x = FADR_Yes xs
      | otherwise   = case go xs of
          FADR_Yes xs' -> FADR_Yes (x :> xs')
          FADR_No      -> FADR_No

size :: forall n a. SizedSet n a -> SNat n
size = coerce (vLength @n @a)
