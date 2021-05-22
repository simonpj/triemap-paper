{- *********************************************************************
*                                                                      *
             Naturals (unsafe, efficient)
*                                                                      *
********************************************************************* -}

{-# LANGUAGE ScopedTypeVariables, TypeApplications, TypeOperators,
             RankNTypes, GADTs, AllowAmbiguousTypes, PolyKinds, DataKinds #-}

module Nat.Unsafe
  ( unsafeAssumeEqual
  , unsafeAssumeSucc
  ) where

import Data.Type.Equality ( (:~:)(..) )
import Unsafe.Coerce

import Prelim

unsafeAssumeEqual :: forall (n :: Nat) (m :: Nat) r. ((n ~ m) => r) -> r
unsafeAssumeEqual k = case unsafeCoerce @(() :~: ()) @(n :~: m) Refl of Refl -> k

data IsSucc n where
  MkIsSucc :: IsSucc (Succ n')

unsafeAssumeSucc :: forall n r. (forall m. (n ~ Succ m) => r) -> r
unsafeAssumeSucc k = case unsafeCoerce @(IsSucc (Succ Zero)) @(IsSucc n) MkIsSucc of
  mk@MkIsSucc -> case mk of (_ :: IsSucc (Succ m)) -> k @m
