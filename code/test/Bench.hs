{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}

import GenTrieMap
import Arbitrary
import Test.QuickCheck.Gen
import Test.QuickCheck.Random
import Control.DeepSeq
import Gauge

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

-- | Run the given 'Gen' starting from a deterministic, fixed seed and the given size parameter.
runGenDet :: Int -> Gen a -> a
runGenDet size gen = unGen gen (mkQCGen 42) size

buildExprMap :: Int -> ExprMap Int
buildExprMap n = mkClosedExprMap $ runGenDet n (vectorOf n genClosedExpr)

main = defaultMain
  [ lookup_all 10
  , lookup_all 100
  , lookup_all 300
  , lookup_all 500
  , lookup_all 1000
  ]
  where
    lookup_all :: Int -> Benchmark
    lookup_all n =
      env (pure (runGenDet n $ vectorOf n genClosedExpr)) $ \exprs ->
      env (pure (mkClosedExprMap exprs)) $ \em ->
      bench (show n) $ nf (map (\e -> lookupTM (closedToDBExpr e) em)) exprs

