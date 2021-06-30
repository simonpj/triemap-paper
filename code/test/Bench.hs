{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}

import GenTrieMap
import Arbitrary

import Test.QuickCheck.Gen
import Test.QuickCheck.Random
import Control.DeepSeq
import Gauge

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import Data.Hashable
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap


{- *********************************************************************
*                                                                      *
               NFData orphans
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

instance (NFData (tm (ListMap tm v)), NFData v) => NFData (ListMap tm v) where
  rnf lm = rnf [rnf (lm_nil lm), rnf (lm_cons lm)]

instance NFData v => NFData (ExprMap' v) where
  rnf EM{..} = rnf [rnf em_bvar, rnf em_fvar, rnf em_app, rnf em_lit, rnf em_lam ]


{- *********************************************************************
*                                                                      *
               Hashable orophan
*                                                                      *
********************************************************************* -}


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

instance MapAPI (ExprMap Int) where
  emptyMap = emptyExprMap
  lookupMap e = lookupTM (deBruijnize e)
  insertMap e = insertTM (deBruijnize e)

instance MapAPI (Map Expr Int) where
  emptyMap = Map.empty
  lookupMap = Map.lookup
  insertMap = Map.insert

instance MapAPI (HashMap Expr Int) where
  emptyMap = HashMap.empty
  lookupMap = HashMap.lookup
  insertMap = HashMap.insert

mapFromList :: MapAPI m => [(Expr, Int)] -> m
mapFromList = foldr (uncurry insertMap) emptyMap


{- *********************************************************************
*                                                                      *
               Benchmarks
*                                                                      *
********************************************************************* -}

sizes :: [Int]
sizes = [100, 500, 1000]

main = defaultMain
  [ bgroup "filler, so that the first line begins with a leading comma" []
  , lookup_all_w_prefix
  , insert_lookup_one
  , lookup_all
  , lookup_one
  ]
  where
    with_map_of_exprs :: forall m. MapAPI m => Int -> ([Expr] -> m -> Benchmark) -> Benchmark
    with_map_of_exprs n k =
      env (pure (runGenDet n $ vectorOf n genClosedExpr)) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      k exprs expr_map

    bench_all_variants :: String -> [Int] -> (forall m. MapAPI m => m -> Int -> Benchmark) -> Benchmark
    bench_all_variants name sizes f = bgroup name $ flip map sizes $ \n -> bgroup (show n)
      [ bgroup "ExprMap" [f (emptyMap :: ExprMap Int)      n]
      , bgroup "Map"     [f (emptyMap :: Map Expr Int)     n]
      , bgroup "HashMap" [f (emptyMap :: HashMap Expr Int) n]
      ]

    lookup_all :: Benchmark
    lookup_all = bench_all_variants "lookup_all" sizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \exprs expr_map ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    lookup_one :: Benchmark
    lookup_one = bench_all_variants "lookup_one" sizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \exprs expr_map ->
      bench "" $ nf (`lookupMap` expr_map) (head exprs) -- exprs is random, so head is as good as any

    insert_lookup_one :: Benchmark
    insert_lookup_one = bench_all_variants "insert_lookup_one" sizes $ \(_ :: m) n ->
      with_map_of_exprs @m n $ \_exprs expr_map ->
      env (pure (runGenDet (2*n) genClosedExpr)) $ \e ->
      bench "" $ nf (\e' -> lookupMap e' (insertMap e' (n+1) expr_map)) e

    lookup_all_w_prefix :: Benchmark
    lookup_all_w_prefix = bench_all_variants "lookup_all_w_prefix" sizes $ \(_ :: m) n ->
      env (pure (map (\e -> iterate (Lam "$") e !! n) $ runGenDet n $ vectorOf n genClosedExpr)) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: m)) $ \(expr_map :: m) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

m :: MapAPI m => Int -> m
m n = mapFromList $ zip (runGenDet n $ vectorOf n genClosedExpr) [0..]
