{- *********************************************************************
*                                                                      *
                   Size-indexed sets (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs, TypeApplications,
             ScopedTypeVariables, TypeOperators, PatternSynonyms,
             StandaloneDeriving, DeriveFoldable #-}

module SizedSet.Unsafe
  ( SizedSet
  , fromVec
  , member
  , isEmptySizedSet
  , findAndDelete, FindAndDeleteResult, pattern FADR_Yes, pattern FADR_No
  , size
  ) where

import Prelim
import Nat.Unsafe
import Vec.Unsafe
import SNat.Unsafe

import Data.Type.Equality
import qualified Data.Set as Set
import Data.Coerce

-- Invariant: no duplicates
type SizedSet :: Nat -> Ty -> Ty
newtype SizedSet n a where
  UnsafeMkSizedSet :: Set.Set a -> SizedSet n a

deriving instance Foldable (SizedSet n)
deriving instance Show a => Show (SizedSet n a)

-- fails when there are duplicates
fromVec :: Ord a => Vec n a -> Maybe (SizedSet n a)
fromVec v
  | has_duplicates = Nothing
  | otherwise      = Just $ UnsafeMkSizedSet set
  where
    set            = foldMap Set.singleton v  -- is there a better way to do this?
    has_duplicates = length set /= length v

member :: Ord a => a -> SizedSet n a -> Bool
member x (UnsafeMkSizedSet s) = Set.member x s

isEmptySizedSet :: forall n a. SizedSet n a -> Maybe (n :~: Zero)
isEmptySizedSet (UnsafeMkSizedSet s)
  | Set.null s = unsafeAssumeEqual @n @Zero (Just Refl)
  | otherwise  = Nothing

type FindAndDeleteResult :: Nat -> Ty -> Ty
data FindAndDeleteResult n a where
  FADR_Yes :: SizedSet n a -> FindAndDeleteResult (Succ n) a
  FADR_No  :: FindAndDeleteResult n a

findAndDelete :: forall n a. Ord a => a -> SizedSet n a -> FindAndDeleteResult n a
findAndDelete needle (UnsafeMkSizedSet s)
  = case Set.alterF f needle s of
      Nothing -> FADR_No
      Just s' -> unsafeAssumeSucc @n $
                 FADR_Yes (UnsafeMkSizedSet s')
  where
    f :: Bool -> Maybe Bool
    f False = Nothing     -- needle not found: fail
    f True  = Just False  -- needle found: remove it

size :: forall n a. SizedSet n a -> SNat n
size = coerce Set.size

