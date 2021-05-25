{- *********************************************************************
*                                                                      *
             Singleton Naturals (safe, inefficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE TypeOperators, DataKinds, StandaloneKindSignatures, GADTs,
             PatternSynonyms #-}

module SNat.Safe
  ( SNat
  , pattern SZero
  , pattern SSucc
  , (%+)
  , SNatI(snat)
  , snatToInt
  , eqSNat
  ) where

import Prelim

import Data.Kind          ( Constraint )
import Data.Type.Equality ( (:~:)(..) )

type SNat :: Nat -> Ty
data SNat n where
  SZero :: SNat Zero
  SSucc :: SNat n -> SNat (Succ n)

(%+) :: SNat n1 -> SNat n2 -> SNat (n1 + n2)
SZero     %+ n2 = n2
SSucc n1' %+ n2 = SSucc (n1' %+ n2)

snatToInt :: SNat n -> Int
snatToInt SZero = 0
snatToInt (SSucc n) = 1 + snatToInt n

type SNatI :: Nat -> Constraint
class SNatI n where
  snat :: SNat n
instance SNatI Zero where
  {-# INLINE snat #-}
  snat = SZero
instance SNatI n => SNatI (Succ n) where
  {-# INLINE snat #-}
  snat = SSucc snat

eqSNat :: SNat m -> SNat n -> Maybe (m :~: n)
eqSNat SZero SZero = Just Refl
eqSNat (SSucc m) (SSucc n) = do
  Refl <- eqSNat m n
  return Refl
eqSNat _ _ = Nothing
