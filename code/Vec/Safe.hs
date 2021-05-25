{- *********************************************************************
*                                                                      *
             Length-indexed vectors
*                                                                      *
********************************************************************* -}

{-# LANGUAGE DataKinds, StandaloneKindSignatures, GADTs, StandaloneDeriving,
             DeriveFunctor, DeriveFoldable, ScopedTypeVariables #-}

module Vec.Safe
  ( Vec
  , pattern Nil
  , pattern (:>)
  , vLength
  , vReplicate
  , (!!!)
  , vUpdateAt
  , vSnoc
  , vZipEqual
  , fins, finsI
  , EVec(..)
  , vecFromList
  ) where

import Prelim
import SNat.Safe
import Fin.Safe

type Vec :: Nat -> Ty -> Ty
data Vec n a where
  Nil  :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a
infixr 5 :>

deriving instance Functor (Vec n)
deriving instance Foldable (Vec n)
deriving instance Show a => Show (Vec n a)

vLength :: Vec n a -> SNat n
vLength Nil       = SZero
vLength (_ :> xs) = SSucc (vLength xs)

vReplicate :: SNat n -> a -> Vec n a
vReplicate SZero     _ = Nil
vReplicate (SSucc n) x = x :> vReplicate n x

(!!!) :: Vec n a -> Fin n -> a
v !!! f = go f v
  where
    go :: Fin n -> Vec n a -> a  -- annoying laziness workaround
    go FZero (x :> _) = x
    go (FSucc f) (_ :> xs) = go f xs

vUpdateAt :: forall n a. Vec n a -> Fin n -> (a -> a) -> Vec n a
vUpdateAt v f upd = go f v
  where
    go :: Fin m -> Vec m a -> Vec m a
    go FZero     (x :> xs) = upd x :> xs
    go (FSucc f) (x :> xs) = x :> go f xs

vSnoc :: Vec n a -> a -> Vec (Succ n) a
vSnoc Nil       y = y :> Nil
vSnoc (x :> xs) y = x :> vSnoc xs y

vZipEqual :: Vec n a -> Vec n b -> Vec n (a,b)
vZipEqual Nil Nil = Nil
vZipEqual (x :> xs) (y :> ys) = (x,y) :> vZipEqual xs ys

fins :: SNat n -> Vec n (Fin n)
fins SZero = Nil
fins (SSucc n) = FZero :> fmap FSucc (fins n)

finsI :: SNatI n => Vec n (Fin n)
finsI = fins snat

type EVec :: Ty -> Ty
data EVec a where
  MkEV :: Vec n a -> EVec a

vecFromList :: [a] -> EVec a
vecFromList [] = MkEV Nil
vecFromList (x : xs) = case vecFromList xs of MkEV xs' -> MkEV (x :> xs')
