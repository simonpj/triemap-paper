{- *********************************************************************
*                                                                      *
                   Maps (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE DataKinds, StandaloneKindSignatures, GADTs #-}

module FinMap.Unsafe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , bumpFinMapIndex
  ) where

import Prelim
import Fin.Unsafe
import SNat.Unsafe

import qualified Data.IntMap as IntMap
import Data.Coerce

type FinMap :: Nat -> Ty -> Ty
newtype FinMap n a where
  UnsafeMkFinMap :: IntMap.IntMap a -> FinMap n a

emptyFinMap :: SNatI n => FinMap n a
emptyFinMap = UnsafeMkFinMap IntMap.empty

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap (UnsafeMkFin k) (UnsafeMkFinMap m) = IntMap.lookup k m

insertFinMap :: Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap (UnsafeMkFin k) val (UnsafeMkFinMap m) = UnsafeMkFinMap (IntMap.insert k val m)

bumpFinMapIndex :: FinMap n a -> FinMap (Succ n) a
bumpFinMapIndex = coerce
