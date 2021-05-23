{- *********************************************************************
*                                                                      *
             Length-indexed vectors (with unsafe Fins)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE DataKinds, StandaloneKindSignatures, GADTs, StandaloneDeriving,
             DeriveFunctor, DeriveFoldable, ScopedTypeVariables, TypeApplications #-}

module Vec.Unsafe
  ( Vec(..)
  , vLength
  , vReplicate
  , (!!!)
  , vUpdateAt
  , vSnoc
  , vZipEqual
  , fins, finsI
  ) where

import Prelim
import SNat.Unsafe
import Fin.Unsafe

import Data.Coerce

type Vec :: Nat -> Ty -> Ty
newtype Vec n a where
  UnsafeMkVec :: [a] -> Vec n a

deriving instance Functor (Vec n)
deriving instance Foldable (Vec n)
deriving instance Show a => Show (Vec n a)

vLength :: forall n a. Vec n a -> SNat n
vLength = coerce (length @[] @a)

vReplicate :: forall n a. SNat n -> a -> Vec n a
vReplicate = coerce (replicate @a)

(!!!) :: forall n a. Vec n a -> Fin n -> a
(!!!) = coerce ((!!) @a)

vUpdateAt :: forall n a. Vec n a -> Fin n -> (a -> a) -> Vec n a
vUpdateAt (UnsafeMkVec v) (UnsafeMkFin index) upd = UnsafeMkVec (go index v)
  where
    go :: Int -> [a] -> [a]
    go _ []       = error "vUpdateAt"
    go 0 (x : xs) = upd x : xs
    go n (x : xs) = x : go (n-1) xs

vSnoc :: Vec n a -> a -> Vec (Succ n) a
vSnoc (UnsafeMkVec list) x = coerce (list ++ [x])

vZipEqual :: forall n a b. Vec n a -> Vec n b -> Vec n (a,b)
vZipEqual = coerce (zip @a @b)

fins :: SNat n -> Vec n (Fin n)
fins n = UnsafeMkVec (go [] (snatToInt n))
  where
    go :: [Fin n] -> Int -> [Fin n]
    go acc 0 = acc
    go acc n = go (UnsafeMkFin (n-1) : acc) (n-1)

finsI :: SNatI n => Vec n (Fin n)
finsI = fins snat
