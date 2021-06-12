{-# LANGUAGE ScopedTypeVariables, TypeApplications, DerivingStrategies,
             GeneralizedNewtypeDeriving #-}

module Bag ( Bag, union, single, empty, fromList, Bag.null ) where

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

null :: Bag a -> Bool
null (MkBag []) = True
null _ = False
