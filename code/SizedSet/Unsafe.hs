{- *********************************************************************
*                                                                      *
                   Size-indexed sets (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs #-}

module SizedSet.Unsafe
  ( SizedSet
  , fromVec
  , member
  ) where

import Prelim
import Vec.Unsafe

import qualified Data.Set as Set

-- Invariant: no duplicates
type SizedSet :: Nat -> Ty -> Ty
newtype SizedSet n a where
  UnsafeMkSizedSet :: Set.Set a -> SizedSet n a

-- fails when there are duplicates
fromVec :: Ord a => Vec n a -> Maybe (SizedSet n a)
fromVec v
  | has_duplicates = Nothing
  | otherwise      = Just $ UnsafeMkSizedSet set
  where
    set            = foldMap Set.singleton v  -- is there a better way to do this?
    has_duplicates = length set == length v

member :: Ord a => a -> SizedSet n a -> Bool
member x (UnsafeMkSizedSet s) = Set.member x s
