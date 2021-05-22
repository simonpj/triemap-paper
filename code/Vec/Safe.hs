{- *********************************************************************
*                                                                      *
             Length-indexed vectors
*                                                                      *
********************************************************************* -}

{-# LANGUAGE DataKinds, StandaloneKindSignatures, GADTs, StandaloneDeriving,
             DeriveFunctor, DeriveFoldable, ScopedTypeVariables #-}

module Vec.Safe
  ( Vec(..)
  , vReplicate
  , (!!!)
  , vUpdateAt
  , vSnoc
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

vReplicate :: SNat n -> a -> Vec n a
vReplicate SZero     _ = Nil
vReplicate (SSucc n) x = x :> vReplicate n x

(!!!) :: Vec n a -> Fin n -> a
v !!! f = go f v
  where
    go :: Fin n -> Vec n a -> a  -- annoying laziness workaround
    go FZero (x :> _) = x
    go (FSucc f) (_ :> xs) = go f xs

vUpdateAt :: forall n a. Vec n a -> Fin n -> a -> Vec n a
vUpdateAt v f y = go f v
  where
    go :: Fin m -> Vec m a -> Vec m a
    go FZero     (_ :> xs) = y :> xs
    go (FSucc f) (x :> xs) = x :> go f xs

vSnoc :: Vec n a -> a -> Vec (Succ n) a
vSnoc Nil       y = y :> Nil
vSnoc (x :> xs) y = x :> vSnoc xs y
