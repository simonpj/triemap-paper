{- *********************************************************************
*                                                                      *
             Singleton Naturals (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeApplications, TypeOperators, ScopedTypeVariables,
             DataKinds, StandaloneKindSignatures, GADTs, PolyKinds,
             PatternSynonyms, ViewPatterns #-}

module SNat.Unsafe
  ( SNat
  , pattern SZero
  , pattern SSucc
  , SNatI(snat)
  , snatToInt
  , eqSNat

  -- * Unsafe exports
  , pattern UnsafeMkSNat
  ) where

import Prelim
import Nat.Unsafe

import Data.Kind          ( Constraint )
import Data.Type.Equality ( (:~:)(..) )

type SNat :: Nat -> Ty
newtype SNat n where
  UnsafeMkSNat :: { snatToInt :: Int } -> SNat n

szeroPat :: forall n. SNat n -> Maybe (n :~: Zero)
szeroPat (UnsafeMkSNat 0) = unsafeAssumeEqual @n @Zero (Just Refl)
szeroPat _                = Nothing

pattern SZero :: forall n. () => (n ~ Zero) => SNat n
pattern SZero <- (szeroPat -> Just Refl)
  where SZero = UnsafeMkSNat 0

data SSuccPatResult n where
  SPR_Yes :: SNat n -> SSuccPatResult (Succ n)
  SPR_No  :: SSuccPatResult n

ssuccPat :: forall n. SNat n -> SSuccPatResult n
ssuccPat (UnsafeMkSNat n)
  | n > 0     = unsafeAssumeSucc @n (SPR_Yes (UnsafeMkSNat (n-1)))
  | otherwise = SPR_No

pattern SSucc :: forall n. () => forall m. (n ~ Succ m) => SNat m -> SNat n
pattern SSucc n <- (ssuccPat -> SPR_Yes n)
  where
    SSucc (UnsafeMkSNat n) = UnsafeMkSNat (1+n)

{-# COMPLETE SZero, SSucc #-}

type SNatI :: Nat -> Constraint
class SNatI n where
  snat :: SNat n
instance SNatI Zero where
  {-# INLINE snat #-}
  snat = UnsafeMkSNat 0
instance SNatI n => SNatI (Succ n) where
  {-# INLINE snat #-}
  snat = UnsafeMkSNat (1 + snatToInt (snat @n))

eqSNat :: forall m n. SNat m -> SNat n -> Maybe (m :~: n)
eqSNat (UnsafeMkSNat m) (UnsafeMkSNat n)
  | m == n    = unsafeAssumeEqual @m @n (Just Refl)
  | otherwise = Nothing
