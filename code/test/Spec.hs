{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

import TrieMap
import Arbitrary

import Data.Maybe
import System.Exit
import Data.Tree.View

import Test.QuickCheck

-- Properties for non-matching ExprMap

prop_ExprMap_empe =
  forAll genClosedExpr $ \e ->
    isNothing $ lookupTM (deBruijnize e) (emptyExprMap :: ExprMap Int)

prop_ExprMap_alter_hit =
  forAll genClosedExpr $ \e ->
  forAll genClosedExprMap $ \m -> do
    let xt = fmap (+1)
    let de = deBruijnize e
    lookupTM de (alterTM de xt m) == xt (lookupTM de m)

prop_ExprMap_alter_miss =
  forAll genClosedExpr $ \e1 ->
  forAll (genClosedExpr `suchThat` (/= e1)) $ \e2 ->
  forAll genClosedExprMap $ \m -> do
    let xt = fmap (+1)
    let de1 = deBruijnize e1
    let de2 = deBruijnize e2
    lookupTM de1 (alterTM de2 xt m) == lookupTM de1 m

-- Properties for PatMap

-- This property ensures that if we get any matches, that they substitute to the
-- actual expr that that we're looking up.
prop_match_hit = withMaxSuccess 10000 $
  forAllShow (genPatSet) (Data.Tree.View.showTree . patMapToTree) $ \ps ->
  let pats = elemsPatSet ps in
  not (null pats) ==> forAll (elements pats) $ \pat ->
  forAll (genInstance pat) $ \e ->
  let matches = matchPM e ps in
  counterexample ("PatMap:\n" ++ Data.Tree.View.showTree (patMapToTree ps)) $
  counterexample ("Pat:  " ++ show pat) $
  counterexample ("Expr: " ++ show e) $
  counterexample ("Matches: " ++ show matches) $
    not (null matches)
     && all (e ==) [ applySubst subst e | (subst, e) <- matches ]

-- This property ensures that after we
prop_match_miss = withMaxSuccess 10000 $
  forAll genPatSet $ \ps ->
  let pats = elemsPatSet ps in
  not (null pats) ==> forAll (elements pats) $ \pat ->
  forAll (genInstance pat) $ \e ->
  let matches = matchPM e ps in
  -- now delete these matches and try again
  let ps' = foldr (\(subst, e) ps -> deletePM (map fst subst) e ps) ps matches in
  counterexample (show pat) $
    null $ matchPM e ps' -- should show now matches

-- | A regression test exemplifying that is_more_specific is broken:
prop_too_specific = do
  let m = mkPatSet [(["a", "b", "c"], read "F (a b) c"), (["ab"], read "F ab C")]
      matches = matchPM (read "F (A B) C") m
  counterexample ("Matching on `F (A B) C` returned wrong number of matches:\n" ++ show matches) $
    length matches == 2

prop_regression_test1 = do
  let e1 = read "\\d. d"
      p1 = ([], e1)
      p2 = (["a"], read "a b")
  let m = mkPatSet [p1, p2]
      matches = matchPM e1 m
  counterexample ("Matching on `id` should return `id`:\n" ++ show matches) $
    all (e1 ==) [ applySubst subst e | (subst, e) <- matches ]

prop_regression_test2 = do
  let p1 = (["a"], read "\\c. a a")
  let p2 = (["a", "b", "c"], read "a b c")
  let m = mkPatSet [p1, p2]
      matches = matchPM (read "λc. (λa.a) (λa.a)") m
  counterexample ("Matching `λc. (λa.a) (λa.a)` against `forall a. λc. a a`:\n" ++ show matches) $
    length matches == 1

prop_regression_test3 =
  let pat = (["a"], read "P a (\\b. a)")
      tar =         read "P (\\c. b) (\\d.\\c. b)" in
  let m = mkPatSet [pat]
      matches = matchPM tar m in
  counterexample ("Pattern:" ++ show pat) $
  counterexample ("Target:" ++ show tar) $
  counterexample ("Probably captured b. Matches:\n" ++ show [ applySubst subst e | (subst, e) <- matches ]) $
    all (tar ==) [ applySubst subst e | (subst, e) <- matches ]

prop_regression_test4 =
  let pat = (["a"], read "(\\b. a)")
      tar =         read "(\\d. b)" in
  let m = mkPatSet [pat]
      matches = matchPM tar m in
  counterexample ("Pattern:" ++ show pat) $
  counterexample ("Target:" ++ show tar) $
  counterexample ("Probably captured b. Matches:\n" ++ show [ applySubst subst e | (subst, e) <- matches ]) $
    all (tar ==) [ applySubst subst e | (subst, e) <- matches ]

--distinctValues :: Eq a => [(x,y,a)] -> Bool
-- Assume that values are Ints to avoid ambiguous exprs
distinctValues :: [(x,y,Int)] -> Bool
distinctValues [] = True
distinctValues ((_,_,a):xs) = notIn a xs && distinctValues xs
  where
    notIn a [] = True
    notIn a ((_,_,b):xs) = a /= b && notIn a xs

return []
main :: IO ()
main = do
  b <- $quickCheckAll
  if b then exitSuccess
       else exitFailure
