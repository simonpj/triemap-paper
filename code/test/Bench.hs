{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}

module Bench ( gauge, weigh ) where

import GenTrieMap
import Arbitrary

import Test.QuickCheck.Gen
import Test.QuickCheck.Random
import Control.DeepSeq
import Gauge
import DataSize

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import Data.Hashable
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap


{- *********************************************************************
*                                                                      *
               orphans
*                                                                      *
********************************************************************* -}


instance NFData Expr where
  rnf (Lit l) = rnf l
  rnf (Var v) = rnf v
  rnf (App f a) = rnf f `seq` rnf a
  rnf (Lam v e) = rnf v `seq` rnf e

instance NFData a => NFData (DeBruijn a) where
  rnf (D env a) = rnf env `seq` rnf a

instance NFData DeBruijnEnv where
  rnf env = rnf (dbe_env env)

instance (TrieMap tm, NFData (TrieKey tm), NFData (tm v), NFData v) => NFData (SEMap tm v) where
  rnf EmptySEM = ()
  rnf (SingleSEM k v) = rnf k `seq` rnf v
  rnf (MultiSEM tm) = rnf tm

instance (NFData (tm (ListMap tm v)), NFData v) => NFData (ListMap' tm v) where
  rnf lm = rnf [rnf (lm_nil lm), rnf (lm_cons lm)]

instance NFData v => NFData (ExprMap' v) where
  rnf EM{..} = rnf [rnf em_bvar, rnf em_fvar, rnf em_app, rnf em_lit, rnf em_lam ]

instance Hashable Expr where
  hashWithSalt salt e = go salt (deBruijnize e)
    where
      go salt (D _ (Lit l))     = salt `hashWithSalt` (0::Int) `hashWithSalt` l
      go salt (D env (App f a)) = salt `hashWithSalt` (1::Int) `go` D env f `go` D env a
      go salt (D env (Lam v e)) = salt `hashWithSalt` (2::Int) `go` D (extendDBE v env) e
      go salt (D env (Var v))   = salt `hashWithSalt` case lookupDBE v env of
        Nothing -> (3::Int) `hashWithSalt` v
        Just bv -> (4::Int) `hashWithSalt` bv


{- *********************************************************************
*                                                                      *
               Map API
*                                                                      *
********************************************************************* -}


-- | Run the given 'Gen' starting from a deterministic, fixed seed and the given size parameter.
runGenDet :: Int -> Gen a -> a
runGenDet size gen = unGen gen (mkQCGen 42) size

class NFData m => MapAPI m where
  emptyMap :: m
  lookupMap :: Expr -> m -> Maybe Int
  insertMap :: Expr -> Int -> m -> m
  fold :: (Int -> b -> b) -> m -> b -> b
  mapFromList :: [(Expr, Int)] -> m

instance MapAPI (ExprMap Int) where
  emptyMap = emptyExprMap
  lookupMap e = lookupTM (deBruijnize e)
  insertMap e = insertTM (deBruijnize e)
  fold = foldTM
  mapFromList = foldr (uncurry insertMap) emptyMap

instance MapAPI (Map Expr Int) where
  emptyMap = Map.empty
  lookupMap = Map.lookup
  insertMap = Map.insert
  fold f m z = foldr f z m
  mapFromList = Map.fromList

instance MapAPI (HashMap Expr Int) where
  emptyMap = HashMap.empty
  lookupMap = HashMap.lookup
  insertMap = HashMap.insert
  fold f m z = foldr f z m
  mapFromList = HashMap.fromList

mkNExprs :: Int -> [Expr]
mkNExprs n = runGenDet 100 $ vectorOf n genClosedExpr
mkNExprsWithPrefix :: Int -> (Expr -> Expr) -> [Expr]
mkNExprsWithPrefix n f = map (\e -> iterate f e !! n) (mkNExprs n)


{- *********************************************************************
*                                                                      *
               Benchmarks
*                                                                      *
********************************************************************* -}

gaugeSizes :: [Int]
-- gaugeSizes = [100, 1000]
gaugeSizes = [10, 100, 1000]
-- gaugeSizes = [10, 100, 1000, 10000]

gauge :: [Benchmark]
gauge =
  [ bgroup "random"
      [ bgroup "filler, so that the first line begins with a leading comma" []
      , rnd_lookup_all
      , rnd_lookup_all_w_app1_prefix
      , rnd_lookup_all_w_app2_prefix
      , rnd_lookup_all_w_lam_prefix
      , rnd_lookup_one
      , rnd_insert_lookup_one
      , rnd_fromList_w_app1_prefix
      , rnd_fromList
      , rnd_fold
      ]
  ]
  where
    with_map_of_exprs :: forall m. MapAPI m => Int -> ([Expr] -> m -> Benchmark) -> Benchmark
    with_map_of_exprs n k =
      env (pure (mkNExprs n)) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      k exprs expr_map

    bench_all_variants :: String -> [Int] -> (forall m. MapAPI m => m -> Int -> Benchmark) -> Benchmark
    bench_all_variants name sizes f = bgroup name $ flip map sizes $ \n -> bgroup (show n)
      [ bgroup "ExprMap" [f (emptyMap :: ExprMap Int)      n]
      , bgroup "Map"     [f (emptyMap :: Map Expr Int)     n]
      , bgroup "HashMap" [f (emptyMap :: HashMap Expr Int) n]
      ]
    {-# INLINE bench_all_variants #-}

    rnd_lookup_all :: Benchmark
    rnd_lookup_all = bench_all_variants "lookup_all" gaugeSizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \exprs expr_map ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_one :: Benchmark
    rnd_lookup_one = bench_all_variants "lookup_one" gaugeSizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \exprs expr_map ->
      bench "" $ nf (`lookupMap` expr_map) (head exprs) -- exprs is random, so head is as good as any

    rnd_insert_lookup_one :: Benchmark
    rnd_insert_lookup_one = bench_all_variants "insert_lookup_one" gaugeSizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \_exprs expr_map ->
      env (pure (runGenDet (2*n) genClosedExpr)) $ \e ->
      bench "" $ nf (\e' -> lookupMap e' (insertMap e' (n+1) expr_map)) e

    rnd_lookup_all_w_lam_prefix :: Benchmark
    rnd_lookup_all_w_lam_prefix = bench_all_variants "lookup_all_w_lam_prefix" gaugeSizes $ \(_ :: m) n ->
      env (pure (mkNExprsWithPrefix n (Lam "$"))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_all_w_app1_prefix :: Benchmark
    rnd_lookup_all_w_app1_prefix = bench_all_variants "lookup_all_w_app1_prefix" gaugeSizes $ \(_ :: m) n ->
      env (pure (mkNExprsWithPrefix n (Lit "$" `App`))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_all_w_app2_prefix :: Benchmark
    rnd_lookup_all_w_app2_prefix = bench_all_variants "lookup_all_w_app2_prefix" gaugeSizes $ \(_ :: m) n ->
      env (pure (mkNExprsWithPrefix n (`App` Lit "$"))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_fromList :: Benchmark
    rnd_fromList = bench_all_variants "fromList" gaugeSizes $ \(_ :: m) n ->
      env (pure (flip zip [0..] $ mkNExprs n)) $ \pairs ->
      bench "" $ nf (mapFromList :: [(Expr, Int)] -> m) pairs

    rnd_fromList_w_app1_prefix :: Benchmark
    rnd_fromList_w_app1_prefix = bench_all_variants "fromList_w_app1_prefix" gaugeSizes $ \(_ :: m) n ->
      env (pure (flip zip [0..] $ mkNExprsWithPrefix n (Lit "$" `App`))) $ \pairs ->
      bench "" $ nf (mapFromList :: [(Expr, Int)] -> m) pairs

    rnd_fold :: Benchmark
    rnd_fold = bench_all_variants "fold" gaugeSizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \_exprs expr_map ->
      bench "" $ whnf (\em -> fold (+) em 0) expr_map

-- No unionTM yet...
--    rnd_union :: Benchmark
--    rnd_union = bench_all_variants "rnd_union" gaugeSizes $ \(_ :: m) n ->
--      with_map_of_exprs @m n $ \_exprs1 expr_map1 ->
--      with_map_of_exprs @m (n+1) $ \_exprs2 expr_map2 ->
--      bench "" $ nf (map (`union` expr_map)) exprs

weighSizes :: [Int]
weighSizes = [10, 100, 1000]
-- weighSizes = [10, 100, 1000, 5000]

-- Called weigh, because I started out by using the Weigh benchmark framework,
-- but it delivered incorrect results. Still like the name
weigh :: IO ()
weigh = do
  rnd
  rnd_w_lam_prefix
  rnd_w_app1_prefix
  rnd_w_app2_prefix
  where
    weigh_all_variants :: String -> [Int] -> (forall m. MapAPI m => String -> m -> Int -> IO ()) -> IO ()
    weigh_all_variants pref sizes f = flip mapM_ sizes $ \n -> do
      f (pref ++ "/ExprMap") (emptyMap :: ExprMap Int)      n
      f (pref ++ "/Map")     (emptyMap :: Map Expr Int)     n
      f (pref ++ "/HashMap") (emptyMap :: HashMap Expr Int) n

    rnd :: IO ()
    rnd = weigh_all_variants "rnd" weighSizes $ \pref (_ :: m) n -> do
      let map = mapFromList (flip zip [0..] $ mkNExprs n) :: m
      s <- recursiveSize $!! map
      putStrLn (pref ++ "/" ++ show n ++ ": " ++ show s)

    rnd_w_lam_prefix :: IO ()
    rnd_w_lam_prefix = weigh_all_variants "rnd_w_lam_prefix" weighSizes $ \pref (_ :: m) n -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n (Lam "$")) :: m
      s <- recursiveSize $!! map
      putStrLn (pref ++ "/" ++ show n ++ ": " ++ show s)

    rnd_w_app1_prefix :: IO ()
    rnd_w_app1_prefix = weigh_all_variants "rnd_w_app1_prefix" weighSizes $ \pref (_ :: m) n -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n (Lit "$" `App`)) :: m
      s <- recursiveSize $!! map
      putStrLn (pref ++ "/" ++ show n ++ ": " ++ show s)

    rnd_w_app2_prefix :: IO ()
    rnd_w_app2_prefix = weigh_all_variants "rnd_w_app2_prefix" weighSizes $ \pref (_ :: m) n -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n (`App` Lit "$")) :: m
      s <- recursiveSize $!! map
      putStrLn (pref ++ "/" ++ show n ++ ": " ++ show s)

m :: MapAPI m => Int -> m
m n = mapFromList $ zip (runGenDet n $ vectorOf n genClosedExpr) [0..]


