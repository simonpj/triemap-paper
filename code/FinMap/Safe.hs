{- *********************************************************************
*                                                                      *
                   Maps
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs,
             StandaloneDeriving #-}

module FinMap.Safe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , alterFinMap
  , bumpFinMapIndex
  , finMapToList
  ) where

import Prelim
import SNat.Safe
import Fin.Safe
import Vec.Safe

import Data.Foldable ( toList )

type FinMap :: Nat -> Ty -> Ty
newtype FinMap n a where
  MkFinMap :: Vec n (Maybe a) -> FinMap n a

deriving instance Show a => Show (FinMap n a)

emptyFinMap :: SNatI n => FinMap n a
emptyFinMap = MkFinMap (vReplicate snat Nothing)

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap index (MkFinMap v) = v !!! index

insertFinMap :: Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap index elt (MkFinMap v) = MkFinMap (vUpdateAt v index (const $ Just elt))

alterFinMap :: (Maybe a -> Maybe a) -> Fin n -> FinMap n a -> FinMap n a
alterFinMap upd f (MkFinMap v) = MkFinMap (vUpdateAt v f upd)

bumpFinMapIndex :: FinMap n a -> FinMap (Succ n) a
bumpFinMapIndex (MkFinMap v) = MkFinMap (v `vSnoc` Nothing)

finMapToList :: FinMap n a -> [(Fin n, a)]
finMapToList (MkFinMap v) = [ (f, x)
                            | (f, Just x) <- toList $ fins (vLength v) `vZipEqual` v ]
