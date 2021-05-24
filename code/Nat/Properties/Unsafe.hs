{- *********************************************************************
*                                                                      *
             Naturals (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeOperators, DataKinds, GADTs, RankNTypes, AllowAmbiguousTypes #-}

module Nat.Properties.Unsafe
  ( zeroIsRightIdentity
  , succOnRight
  , plusIsAssoc
  ) where

import Prelim
import SNat.Unsafe

import Data.Type.Equality
import Unsafe.Coerce

zeroIsRightIdentity :: SNat n -> n + Zero :~: n
zeroIsRightIdentity _ = unsafeCoerce Refl

succOnRight :: forall m0 n. SNat n -> forall m. (m ~ m0) => n + Succ m :~: Succ (n + m)
succOnRight _ = unsafeCoerce Refl

plusIsAssoc :: forall b0 c0 a. SNat a -> forall b c. (b ~ b0, c ~ c0) => (a + (b + c)) :~: ((a + b) + c)
plusIsAssoc _ = unsafeCoerce Refl
