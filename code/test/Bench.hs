{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

import GenTrieMap
import Arbitrary
import Test.QuickCheck.Gen
import Test.QuickCheck.Random
import Control.DeepSeq
import Gauge
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map


--
-- NFData orphans
--

instance NFData Expr where
  rnf (Lit l) = rnf l
  rnf (Var v) = rnf v
  rnf (App f a) = rnf f `seq` rnf a
  rnf (Lam v e) = rnf v `seq` rnf e

instance NFData ClosedExpr where
  rnf (ClosedExpr e) = rnf e

instance NFData a => NFData (DeBruijn a) where
  rnf (D env a) = rnf env `seq` rnf a

instance NFData DeBruijnEnv where
  rnf env = rnf (dbe_env env)

instance (TrieMap tm, NFData (TrieKey tm), NFData (tm v), NFData v) => NFData (SEMap tm v) where
  rnf EmptySEM = ()
  rnf (SingleSEM k v) = rnf k `seq` rnf v
  rnf (MultiSEM tm) = rnf tm

instance NFData v => NFData (ExprMap' v) where
  rnf em = rnf [rnf (em_bvar em), rnf (em_fvar em), rnf (em_app em), rnf (em_lit em), rnf (em_lam em) ]

--
-- Ord orphans
--

exprTag :: Expr -> Int
exprTag Var{} = 0
exprTag App{} = 1
exprTag Lam{} = 2
exprTag Lit{} = 3
{-# INLINE exprTag #-}

-- | Yuck
instance Ord ClosedExpr where
  compare (ClosedExpr a) (ClosedExpr b) = go (deBruijnize a) (deBruijnize b)
    where
      go (D _ (Lit l1))       (D _ (Lit l2))       = compare l1 l2
      go (D env1 (App f1 a1)) (D env2 (App f2 a2)) = go (D env1 f1) (D env2 f2) <> go (D env1 a1) (D env2 a2)
      go (D env1 (Lam v1 e1)) (D env2 (Lam v2 e2)) = go (D (extendDBE v1 env1) e1) (D (extendDBE v2 env2) e2)
      go (D env1 (Var v1))    (D env2 (Var v2))    = case (lookupDBE v1 env1, lookupDBE v2 env2) of
        (Just bvi1, Just bvi2) -> compare bvi1 bvi2
        (Nothing,   Nothing)   -> compare v1 v2
        (Just _,    Nothing)   -> GT
        (Nothing,   Just _)    -> LT
      go (D _ e1) (D _ e2) = compare (exprTag e1) (exprTag e2)

-- | Run the given 'Gen' starting from a deterministic, fixed seed and the given size parameter.
runGenDet :: Int -> Gen a -> a
runGenDet size gen = unGen gen (mkQCGen 42) size

class NFData m => MapAPI m where
  emptyMap :: m
  lookupMap :: ClosedExpr -> m -> Maybe Int
  insertMap :: ClosedExpr -> Int -> m -> m

instance MapAPI (ExprMap Int) where
  emptyMap = emptyExprMap
  lookupMap e = lookupTM (closedToDBExpr e)
  insertMap e = insertTM (closedToDBExpr e)

instance MapAPI (Map ClosedExpr Int) where
  emptyMap = Map.empty
  lookupMap = Map.lookup
  insertMap = Map.insert

mapFromList :: MapAPI m => [(ClosedExpr, Int)] -> m
mapFromList = foldr (uncurry insertMap) emptyMap

main = defaultMain
  [ bgroup "ExprMap"
     [ lookup_all @(ExprMap Int)
     ]
  , bgroup "Map"
     [ lookup_all @(Map ClosedExpr Int)
     ]
  ]
  where
    lookup_all :: forall m. MapAPI m => Benchmark
    lookup_all = bgroup "lookup_all" $ flip map [100, 300, 500, 1000] $ \n ->
      env (pure (runGenDet n $ vectorOf n genClosedExpr)) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(em :: m) ->
      bench (show n) $ nf (map (`lookupMap` em)) exprs
      -- bench (show n) $ nf (`lookupMap` em) (head exprs)

m :: MapAPI m => Int -> m
m n = mapFromList $ zip (runGenDet n $ vectorOf n genClosedExpr) [0..]
