{- *********************************************************************
*                                                                      *
             Naturals (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeOperators, DataKinds, GADTs, RankNTypes, AllowAmbiguousTypes,
             TypeApplications, ScopedTypeVariables #-}

module Nat.Properties.Safe
  ( zeroIsRightIdentity
  , succOnRight
  , plusIsAssoc
  ) where

import Prelim
import SNat.Safe

import Data.Type.Equality

zeroIsRightIdentity :: SNat n -> n + Zero :~: n
zeroIsRightIdentity SZero = Refl
zeroIsRightIdentity (SSucc n') = gcastWith (zeroIsRightIdentity n') Refl

succOnRight :: forall m0 n. SNat n -> forall m. (m ~ m0) => n + Succ m :~: Succ (n + m)
succOnRight SZero = Refl
succOnRight (SSucc n') = gcastWith (succOnRight n' @m0) Refl

plusIsAssoc :: forall b0 c0 a. SNat a -> forall b c. (b ~ b0, c ~ c0) => (a + (b + c)) :~: ((a + b) + c)
plusIsAssoc SZero = Refl
plusIsAssoc (SSucc a') = gcastWith (plusIsAssoc a' @b0 @c0) $ Refl
