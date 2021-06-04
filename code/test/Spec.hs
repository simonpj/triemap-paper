{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

-- module Spec where -- because it's also our main file

import GenTrieMap
import Arbitrary

import Data.Maybe
import System.Exit

import Test.QuickCheck

insertUC :: forall a. (Env, Type, a) -> TypeMap a -> TypeMap a
insertUC (env, ty, a) = insertTypeMap (boundVars env) ty a

applyMatches :: Eq a => [ ([(TmplVar,Type)], a) ] -> [(Env, Type, a)] -> [Type]
applyMatches matches inputs = [ applySubst subst ty | (subst, a)  <- matches, (env, ty, b) <- inputs, a == b ]

-- This property ensures that if we get any matches, that they substitute to the actual type that
-- that we're looking up.
-- This property can be trivially fulfilled by not returning any matches.
prop_match
  = forAll genInputs $ \inputs ->
    forAll genClosedType $ \ty ->
    distinctValues inputs ==>
    let trie = foldr insertUC emptyTypeMap inputs
        matches = lookupTypeMap ty trie
    in counterexample (show trie) $
       all (ty `alphaEq`) (applyMatches matches inputs)


-- This property ensures that we actually can find things in the trie
prop_find =
  forAll genClosedType $ \ty ->
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
alphaEq ty1 ty2 = eqDeBT (deBruijnize ty1) (deBruijnize ty2)

applySubst :: [(TmplVar, Type)] -> Type -> Type
applySubst subst ty@TyConTy{}    = ty
applySubst subst ty@(TyVarTy tv) = fromMaybe ty $ lookup tv subst
applySubst subst (FunTy arg res) = FunTy (applySubst subst arg) (applySubst subst res)
applySubst subst (ForAllTy tv body) = ForAllTy tv (applySubst (del tv subst) body)
  where
    del k' = filter (\(k,_v) -> k /= k')

genInputs :: Gen [(Env, Type, Int)]
genInputs = listOf $ do
  env <- genEnv
  ty <- genOpenType env
  val <- arbitrary
  pure (env, ty, val)

generalization :: Type -> Gen ([TyVar], Type)
generalization = undefined

return []
main :: IO ()
main = do
  b <- $quickCheckAll
  if b then exitSuccess
       else exitFailure
