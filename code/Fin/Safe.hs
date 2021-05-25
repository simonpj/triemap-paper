{- *********************************************************************
*                                                                      *
             Finite numbers (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeApplications, ScopedTypeVariables, DataKinds,
             StandaloneKindSignatures, GADTs, StandaloneDeriving,
             PatternSynonyms #-}

module Fin.Safe
  ( Fin
  , pattern FZero
  , pattern FSucc
  , finToInt
  , maxFin, maxFinI
  , bumpFinIndex
  ) where

import Data.Coerce
import qualified Data.IntMap as IntMap

import Prelim
import SNat.Safe

type Fin :: Nat -> Ty     -- numbers in the range [0, n)
data Fin n where
  FZero :: Fin (Succ n)
  FSucc :: Fin n -> Fin (Succ n)

deriving instance Show (Fin n)
deriving instance Eq (Fin n)

finToInt :: Fin n -> Int
finToInt FZero = 0
finToInt (FSucc f) = 1 + finToInt f

-- NB: the result is Fin (Succ m), not Fin m
maxFin :: SNat m -> Fin (Succ m)
maxFin SZero = FZero
maxFin (SSucc n) = FSucc (maxFin n)

maxFinI :: forall n. SNatI n => Fin (Succ n)
maxFinI = maxFin (snat @n)

bumpFinIndex :: Fin n -> Fin (Succ n)
bumpFinIndex FZero = FZero
bumpFinIndex (FSucc f) = FSucc (bumpFinIndex f)
