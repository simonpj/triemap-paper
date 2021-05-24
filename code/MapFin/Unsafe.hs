{- *********************************************************************
*                                                                      *
             Map from a key to a Fin (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs, ScopedTypeVariables,
             TypeApplications #-}

module MapFin.Unsafe
  ( MapFin
  , emptyMapFin
  , lookupMapFin
  , insertKeyMapFin
  , mapFinSize
  ) where

import Prelim
import Vec.Unsafe
import Fin.Unsafe
import SNat.Unsafe

import qualified Data.Map as Map
import Data.Coerce

type MapFin :: Ty -> Nat -> Ty
newtype MapFin k n where
  UnsafeMkMapFin :: Map.Map k (Fin n) -> MapFin k n

emptyMapFin :: forall k. MapFin k Zero
emptyMapFin = coerce (Map.empty @k @(Fin Zero))

lookupMapFin :: forall k n. Ord k => k -> MapFin k n -> Maybe (Fin n)
lookupMapFin = coerce (Map.lookup @k @(Fin n))

insertKeyMapFin :: Ord k => k -> MapFin k n -> MapFin k (Succ n)
insertKeyMapFin k mf@(UnsafeMkMapFin m)
  = UnsafeMkMapFin (Map.insert k (maxFin (mapFinSize mf)) (coerce m))

mapFinSize :: forall k n. MapFin k n -> SNat n
mapFinSize = coerce (Map.size @k @(Fin n))
