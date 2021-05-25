{- *********************************************************************
*                                                                      *
             Map from a key to a Fin (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs, ScopedTypeVariables #-}

module MapFin.Safe
  ( MapFin
  , emptyMapFin
  , lookupMapFin
  , insertKeyMapFin
  , mapFinSize
  , mapFinToList
  ) where

import Prelim
import Vec.Safe
import Fin.Safe
import SNat.Safe

import Data.Foldable ( toList )

type MapFin :: Ty -> Nat -> Ty
newtype MapFin k n where
  MkMapFin :: Vec n k -> MapFin k n

emptyMapFin :: MapFin k Zero
emptyMapFin = MkMapFin Nil

lookupMapFin :: forall k n. Ord k => k -> MapFin k n -> Maybe (Fin n)
lookupMapFin k (MkMapFin v) = go v
  where
    go :: Vec m k -> Maybe (Fin m)
    go Nil = Nothing
    go (x :> xs)
      | k == x    = Just FZero
      | otherwise = FSucc <$> go xs

insertKeyMapFin :: Ord k => k -> MapFin k n -> MapFin k (Succ n)
insertKeyMapFin k (MkMapFin v) = MkMapFin (v `vSnoc` k)

mapFinSize :: forall k n. MapFin k n -> SNat n
mapFinSize (MkMapFin v) = vLength v

mapFinToList :: MapFin k n -> [(k, Fin n)]
mapFinToList (MkMapFin v) = toList (v `vZipEqual` fins (vLength v))
