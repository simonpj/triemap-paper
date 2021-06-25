{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

-- module Spec where -- because it's also our main file

import GenTrieMap
import Arbitrary

import Data.Maybe
import System.Exit

import Test.QuickCheck

-- Properties for non-matching ExprMap

prop_ExprMap_empty =
  forAll genClosedExpr $ \e ->
    isNothing $ lookupTM (closedToDBExpr e) (emptyExprMap :: ExprMap Int)

prop_ExprMap_alter_hit =
  forAll genClosedExpr $ \e ->
  forAll genClosedExprMap $ \m -> do
    let xt = fmap (+1)
    let de = closedToDBExpr e
    lookupTM de (alterTM de xt m) == xt (lookupTM de m)

prop_ExprMap_alter_nonhit =
  forAll genClosedExpr $ \e1 ->
  forAll (genClosedExpr `suchThat` (/= e1)) $ \e2 ->
  forAll genClosedExprMap $ \m -> do
    let xt = fmap (+1)
    let de1 = closedToDBExpr e1
    let de2 = closedToDBExpr e2
    lookupTM de1 (alterTM de2 xt m) == xt (lookupTM de1 m)

{-
Commented out until we can do matching again

insertUC :: forall a. (Env, Expr, a) -> ExprMap a -> ExprMap a
insertUC (env, ty, a) = insertMExprMap (boundVars env) ty a

applyMatches :: Eq a => [ ([(PatVar,Expr)], a) ] -> [(Env, Expr, a)] -> [Expr]
applyMatches matches inputs = [ applySubst subst ty | (subst, a)  <- matches, (env, ty, b) <- inputs, a == b ]

-- This property ensures that if we get any matches, that they substitute to the actual type that
-- that we're looking up.
-- This property can be trivially fulfilled by not returning any matches.
prop_match
  = forAll genInputs $ \inputs ->
    forAll genClosedExpr $ \ty ->
    distinctValues inputs ==>
    let trie = foldr insertUC emptyExprMap inputs
        matches = lookupTM ty trie
    in counterexample (show trie) $
       all (ty `eqDBExpr`) (applyMatches matches inputs)


-- This property ensures that we actually can find things in the trie
prop_find =
  forAll genClosedExpr $ \ty ->
  forAll (generalization ty) $ \(tvs, typ) ->
  let [(subst, ())] = lookupTM ty (insertTM tvs typ () emptyTM)
  in ty `eqDBExpr` applySubst subst typ


--distinctValues :: Eq a => [(x,y,a)] -> Bool
-- Assume that values are Ints to avoid ambiguous types
distinctValues :: [(x,y,Int)] -> Bool
distinctValues [] = True
distinctValues ((_,_,a):xs) = notIn a xs && distinctValues xs
  where
    notIn a [] = True
    notIn a ((_,_,b):xs) = a /= b && notIn a xs
-}

applySubst :: [(PatVar, Expr)] -> Expr -> Expr
applySubst subst e@Lit{}   = e
applySubst subst e@(Var v) = fromMaybe e $ lookup v subst
applySubst subst (App arg res) = App (applySubst subst arg) (applySubst subst res)
applySubst subst (Lam v body) = Lam v (applySubst (del v subst) body)
  where
    del k' = filter (\(k,_v) -> k /= k')

genInputs :: Gen [(Env, Expr, Int)]
genInputs = listOf $ do
  env <- genEnv
  ty <- genOpenExpr env
  val <- arbitrary
  pure (env, ty, val)

generalization :: Expr -> Gen ([Var], Expr)
generalization = undefined

return []
main :: IO ()
main = do
  b <- $quickCheckAll
  if b then exitSuccess
       else exitFailure
