{- *********************************************************************
*                                                                      *
                   Maps
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs #-}

module FinMap.Safe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , bumpFinMapIndex
  ) where

import Prelim
import SNat.Safe
import Fin.Safe
import Vec.Safe

type FinMap :: Nat -> Ty -> Ty
newtype FinMap n a where
  MkFinMap :: Vec n (Maybe a) -> FinMap n a

emptyFinMap :: SNatI n => FinMap n a
emptyFinMap = MkFinMap (vReplicate snat Nothing)

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap index (MkFinMap v) = v !!! index

insertFinMap :: Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap index elt (MkFinMap v) = MkFinMap (vUpdateAt v index (Just elt))

bumpFinMapIndex :: FinMap n a -> FinMap (Succ n) a
bumpFinMapIndex (MkFinMap v) = MkFinMap (v `vSnoc` Nothing)
