{- *********************************************************************
*                                                                      *
             Finite numbers (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeApplications, ScopedTypeVariables, DataKinds,
             StandaloneKindSignatures, GADTs, PatternSynonyms,
             ViewPatterns #-}

module Fin.Unsafe
  ( Fin
  , pattern FZero
  , pattern FSucc
  , finToInt
  , maxFin
  , bumpFinIndex

  -- * Unsafe exports below
  , pattern UnsafeMkFin
  ) where

import Data.Coerce
import qualified Data.IntMap as IntMap

import Prelim
import SNat.Unsafe
import Nat.Unsafe

type Fin :: Nat -> Ty     -- numbers in the range [0, n)
newtype Fin n where
  UnsafeMkFin :: { finToInt :: Int } -> Fin n
  deriving (Show, Eq)

data FZeroPatResult n where
  FZPR_Yes :: FZeroPatResult (Succ m)
  FZPR_No  :: FZeroPatResult n

fzeroPat :: forall n. Fin n -> FZeroPatResult n
fzeroPat (UnsafeMkFin 0) = unsafeAssumeSucc @n FZPR_Yes
fzeroPat _ = FZPR_No

pattern FZero :: forall n. () => forall m. (n ~ Succ m) => Fin n
pattern FZero <- (fzeroPat -> FZPR_Yes)
  where FZero = UnsafeMkFin 0

data FSuccPatResult n where
  FSPR_Yes :: Fin m -> FSuccPatResult (Succ m)
  FSPR_No  :: FSuccPatResult n

fsuccPat :: forall n. Fin n -> FSuccPatResult n
fsuccPat (UnsafeMkFin 0) = FSPR_No
fsuccPat (UnsafeMkFin n) = unsafeAssumeSucc @n (FSPR_Yes (UnsafeMkFin (n-1)))

pattern FSucc :: forall n. () => forall m. (n ~ Succ m) => Fin m -> Fin n
pattern FSucc f <- (fsuccPat -> FSPR_Yes f)
  where FSucc (UnsafeMkFin n) = UnsafeMkFin (1 + n)

{-# COMPLETE FZero, FSucc #-}

maxFin :: forall n. SNatI n => Fin (Succ n)
maxFin = UnsafeMkFin (snatToInt (snat @n))

bumpFinIndex :: Fin n -> Fin (Succ n)
bumpFinIndex = coerce
