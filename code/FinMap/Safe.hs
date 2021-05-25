{- *********************************************************************
*                                                                      *
                   Maps
*                                                                      *
********************************************************************* -}

{-# LANGUAGE StandaloneKindSignatures, DataKinds, GADTs,
             StandaloneDeriving, ScopedTypeVariables #-}

module FinMap.Safe
  ( FinMap
  , emptyFinMap
  , lookupFinMap
  , insertFinMap
  , growFinMap
  , alterFinMap
  , finMapToList
  ) where

import Prelim
import SNat.Safe
import Fin.Safe

import Data.Foldable ( toList )
import Control.Arrow ( first )

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

growFinMap :: a -> FinMap n a -> FinMap (Succ n) a
growFinMap elt FMNil = Just elt :>> FMNil
growFinMap elt (x :>> xs) = x :>> growFinMap elt xs

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
