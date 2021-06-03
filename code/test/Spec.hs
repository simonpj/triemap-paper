{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

-- module Spec where -- because it's also our main file

import GenTrieMap
import Arbitrary

import System.Exit

import Test.QuickCheck

insertUC :: forall a. ([TyVar], Type, a) -> TypeMap a -> TypeMap a
insertUC (tvs, ty, a) = insertTypeMap tvs ty a

applyMatches :: Eq a => [ ([(TmplVar,Type)], a) ] -> [([TyVar], Type, a)] -> [Type]
applyMatches matches inputs = [ applySubst subst ty | (subst, a)  <- matches, (tvs, ty, b) <- inputs, a == b ]

-- This property ensures that if we get any matches, that they substitute to the actual type that
-- that we're looking up.
-- This property can be trivially fulfilled by not returning any matches.
prop_match
  = forAll genInputs $ \inputs ->
    forAll genType $ \ty ->
    distinctValues inputs ==>
    let trie = foldr insertUC emptyTypeMap inputs
        matches = lookupTypeMap ty trie
    in all (ty `alphaEq`) (applyMatches matches inputs)

-- This property ensures that we actually can find things in the trie
prop_find =
  forAll genType $ \ty ->
  forAll (generalization ty) $ \(tvs, typ) ->
  let [(subst, ())] = lookupTypeMap ty (insertTypeMap tvs typ () emptyTypeMap)
  in ty `alphaEq` applySubst subst typ


--distinctValues :: Eq a => [(x,y,a)] -> Bool
-- Assume that values are Ints to avoid ambiguous types
distinctValues :: [(x,y,Int)] -> Bool
distinctValues [] = True
distinctValues ((_,_,a):xs) = notIn a xs && distinctValues xs
  where
    notIn a [] = True
    notIn a ((_,_,b):xs) = a /= b && notIn a xs

alphaEq :: Type -> Type -> Bool
alphaEq = undefined

applySubst :: [(TmplVar, Type)] -> Type -> Type
applySubst = undefined

genInputs :: Gen [([TyVar], Type, Int)]
genInputs = undefined

generalization :: Type -> Gen ([TyVar], Type)
generalization = undefined

return []
main :: IO ()
main = do
  b <- $quickCheckAll
  if b then exitSuccess
       else exitFailure
