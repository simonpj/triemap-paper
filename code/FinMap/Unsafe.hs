{- *********************************************************************
*                                                                      *
                   Maps (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE DataKinds, StandaloneKindSignatures, GADTs, ScopedTypeVariables,
             TypeApplications, StandaloneDeriving #-}

module FinMap.Unsafe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , alterFinMap
  , finMapToList

  -- * CompleteFinMap
  , CompleteFinMap
  , emptyCompleteFinMap
  , lookupCompleteFinMap
  , growCompleteFinMap
  , completeFinMapToList
  ) where

import Prelim
import Fin.Unsafe
import SNat.Unsafe

import qualified Data.IntMap as IntMap
import Data.Coerce

type FinMap :: Nat -> Ty -> Ty
newtype FinMap n a where
  UnsafeMkFinMap :: IntMap.IntMap a -> FinMap n a

deriving instance Show a => Show (FinMap n a)

emptyFinMap :: FinMap n a
emptyFinMap = UnsafeMkFinMap IntMap.empty

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap (UnsafeMkFin k) (UnsafeMkFinMap m) = IntMap.lookup k m

insertFinMap :: Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap (UnsafeMkFin k) val (UnsafeMkFinMap m) = UnsafeMkFinMap (IntMap.insert k val m)

alterFinMap :: forall n a. (Maybe a -> Maybe a) -> Fin n -> FinMap n a -> FinMap n a
alterFinMap = coerce (IntMap.alter @a)

finMapToList :: forall n a. FinMap n a -> [(Fin n, a)]
finMapToList = coerce (IntMap.toList @a)

{- *********************************************************************
*                                                                      *
           Complete FinMaps map every possible Fin to a value
*                                                                      *
********************************************************************* -}

type CompleteFinMap :: Nat -> Ty -> Ty
newtype CompleteFinMap n a where
  UnsafeMkCompleteFinMap :: IntMap.IntMap a -> CompleteFinMap n a

deriving instance Show a => Show (CompleteFinMap n a)

emptyCompleteFinMap :: forall a. CompleteFinMap Zero a
emptyCompleteFinMap = UnsafeMkCompleteFinMap IntMap.empty

lookupCompleteFinMap :: Fin n -> CompleteFinMap n a -> a
lookupCompleteFinMap index (UnsafeMkCompleteFinMap m) = m IntMap.! (coerce index)

-- adds a new entry at the end
growCompleteFinMap :: a -> CompleteFinMap n a -> CompleteFinMap (Succ n) a
growCompleteFinMap elt (UnsafeMkCompleteFinMap m)
  = UnsafeMkCompleteFinMap (IntMap.insert k elt m)
  where
    k = IntMap.size m

completeFinMapToList :: forall n a. CompleteFinMap n a -> [(Fin n, a)]
completeFinMapToList = coerce (IntMap.toList @a)
