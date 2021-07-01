{-# LANGUAGE ScopedTypeVariables, TypeApplications, DerivingStrategies,
             GeneralizedNewtypeDeriving #-}

module Bag ( Bag, Bag.union, single, empty, fromList, toList
           , Bag.null, Bag.concatMap, Bag.any ) where

import Data.List as List
import Data.Coerce

newtype Bag a = MkBag [a]
  deriving newtype Functor

union :: forall a. Bag a -> Bag a -> Bag a
union = coerce ((++) @a)

single :: a -> Bag a
single x = coerce [x]

empty :: Bag a
empty = MkBag []

fromList :: [a] -> Bag a
fromList = MkBag

toList :: Bag a -> [a]
toList = coerce

null :: Bag a -> Bool
null (MkBag []) = True
null _ = False

concatMap :: (a -> Bag b) -> Bag a -> Bag b
concatMap f (MkBag xs) = MkBag (List.concatMap (coerce . f) xs)

any :: (a -> Bool) -> Bag a -> Bool
any f (MkBag xs) = List.any f xs