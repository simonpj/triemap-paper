{- *********************************************************************
*                                                                      *
                   Size-indexed sets (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs #-}

module SizedSet.Safe
  ( SizedSet
  , fromVec
  , member
  ) where

import Prelim
import Vec.Safe

import qualified Data.Set as Set

-- Invariant: no duplicates
type SizedSet :: Nat -> Ty -> Ty
newtype SizedSet n a where
  MkSizedSet :: Vec n a -> SizedSet n a

-- fails when there are duplicates
fromVec :: Ord a => Vec n a -> Maybe (SizedSet n a)
fromVec v
  | has_duplicates = Nothing
  | otherwise      = Just $ MkSizedSet v
  where
    has_duplicates = length (foldMap Set.singleton v) == length v

member :: Ord a => a -> SizedSet n a -> Bool
member x (MkSizedSet v) = elem x v
