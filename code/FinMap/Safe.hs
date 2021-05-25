{- *********************************************************************
*                                                                      *
                   Maps
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs,
             StandaloneDeriving, ScopedTypeVariables, TypeApplications #-}

module FinMap.Safe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , alterFinMap
  , finMapToList

  -- * CompleteFinMap
  , CompleteFinMap
  , emptyCompleteFinMap
  , lookupCompleteFinMap
  , growCompleteFinMap
  , completeFinMapToList
  ) where

import Prelim
import SNat.Safe
import Fin.Safe
import Vec.Safe

import Data.Foldable ( toList )
import Control.Arrow ( first )
import Data.Coerce

type FinMap :: Nat -> Ty -> Ty
-- could just store a vector, but then we need SNatI n when constructing
-- the empty FinMap, which is annoying
data FinMap n a where
  FMNil :: FinMap n a
  (:>>) :: Maybe a -> FinMap n a -> FinMap (Succ n) a
infixr 5 :>>

deriving instance Show a => Show (FinMap n a)

emptyFinMap :: FinMap n a
emptyFinMap = FMNil

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap _ FMNil = Nothing
lookupFinMap FZero (x :>> _) = x
lookupFinMap (FSucc f) (_ :>> xs) = lookupFinMap f xs

insertFinMap :: forall n a. Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap index elt m = alterFinMap (\_ -> Just elt) index m

alterFinMap :: (Maybe a -> Maybe a) -> Fin n -> FinMap n a -> FinMap n a
alterFinMap upd index FMNil = buildFinMap (upd Nothing) index
alterFinMap upd FZero (x :>> xs) = upd x :>> xs
alterFinMap upd (FSucc index) (x :>> xs) = x :>> alterFinMap upd index xs

finMapToList :: FinMap n a -> [(Fin n, a)]
finMapToList FMNil = []
finMapToList (Just x  :>> xs) = (FZero, x) : map (first FSucc) (finMapToList xs)
finMapToList (Nothing :>> xs) = map (first FSucc) (finMapToList xs)

-- internal only; inserts the element at the index provided
buildFinMap :: Maybe a -> Fin n -> FinMap n a
buildFinMap elt FZero = elt :>> FMNil
buildFinMap elt (FSucc f) = Nothing :>> buildFinMap elt f

{- *********************************************************************
*                                                                      *
           Complete FinMaps map every possible Fin to a value
*                                                                      *
********************************************************************* -}

type CompleteFinMap :: Nat -> Ty -> Ty
newtype CompleteFinMap n a where
  MkCompleteFinMap :: Vec n a -> CompleteFinMap n a

deriving instance Show a => Show (CompleteFinMap n a)

emptyCompleteFinMap :: forall a. CompleteFinMap Zero a
emptyCompleteFinMap = coerce (Nil @a)

lookupCompleteFinMap :: Fin n -> CompleteFinMap n a -> a
lookupCompleteFinMap index (MkCompleteFinMap v) = v !!! index

growCompleteFinMap :: forall n a. a -> CompleteFinMap n a -> CompleteFinMap (Succ n) a
growCompleteFinMap = coerce (flip (vSnoc @n @a))

completeFinMapToList :: CompleteFinMap n a -> [(Fin n, a)]
completeFinMapToList (MkCompleteFinMap v) = toList (fins (vLength v) `vZipEqual` v)
