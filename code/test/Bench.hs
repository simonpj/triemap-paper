{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}

module Bench where

import TrieMap
import Arbitrary

import Test.QuickCheck.Gen
import Test.QuickCheck.Random
import Control.DeepSeq
import Criterion
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
  rnf (Var v) = rnf v
  rnf (App f a) = rnf f `seq` rnf a
  rnf (Lam v e) = rnf v `seq` rnf e

instance NFData a => NFData (ModAlpha a) where
  rnf (A env a) = rnf env `seq` rnf a

instance NFData DeBruijnEnv where
  rnf env = rnf (dbe_env env)

instance NFData a => NFData (Pat a) where
  rnf (P env a) = rnf env `seq` rnf a

instance NFData PatVarEnv where
  rnf (PVE dbe caps) = rnf dbe `seq` rnf caps

instance (TrieMap tm, TrieKey tm ~ e, NFData e, NFData (tm v), NFData v) => NFData (SEMap e tm v) where
  rnf EmptySEM = ()
  rnf (SingleSEM k v) = rnf k `seq` rnf v
  rnf (MultiSEM tm) = rnf tm

instance (NFData (tm (ListMap tm v)), NFData v) => NFData (ListMap' tm v) where
  rnf lm = rnf (lm_nil lm) `seq` rnf (lm_cons lm)

instance NFData v => NFData (ExprMap' v) where
  rnf EM{..} = rnf em_bvar `seq` rnf em_fvar `seq` rnf em_app `seq` rnf em_lam

instance Hashable Expr where
  hashWithSalt salt e = go salt (deBruijnize e)
    where
      go salt (A env (App f a)) = salt `hashWithSalt` (1::Int) `go` A env f `go` A env a
      go salt (A env (Lam v e)) = salt `hashWithSalt` (2::Int) `go` A (extendDBE v env) e
      go salt (A env (Var v))   = salt `hashWithSalt` case lookupDBE v env of
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
  unionMap :: m -> m -> m
  fold :: (Int -> b -> b) -> m -> b -> b
  mapFromList :: [(Expr, Int)] -> m

instance MapAPI (ExprMap Int) where
  emptyMap = emptyExprMap
  lookupMap e = lookupTM (deBruijnize e)
  insertMap e = insertTM (deBruijnize e)
  unionMap = unionWithTM (\l r -> r)
  fold = foldTM
  {-# NOINLINE fold #-}
  mapFromList = foldr (uncurry insertMap) emptyMap
  -- mapFromList = fromListWithTM last . map (\(k,v) -> (deBruijnize k, v))
    -- slower, last time I checked

instance MapAPI (Map Expr Int) where
  emptyMap = Map.empty
  lookupMap = Map.lookup
  insertMap = Map.insert
  unionMap = Map.unionWith (\l r -> r)
  fold f m z = foldr f z m
  {-# NOINLINE fold #-} -- disable specialisation for f as foldTM can't do it. Apples to apples
  mapFromList = Map.fromList

instance MapAPI (HashMap Expr Int) where
  emptyMap = HashMap.empty
  lookupMap = HashMap.lookup
  insertMap = HashMap.insert
  unionMap = HashMap.unionWith (\l r -> r)
  fold f m z = foldr f z m
  {-# NOINLINE fold #-} -- disable specialisation for f as foldTM can't do it. Apples to apples
  mapFromList = HashMap.fromList

mkNExprs :: Int -> Int -> [Expr]
mkNExprs map_size expr_size = runGenDet expr_size $ vectorOf map_size genClosedExpr
mkNExprsWithPrefix :: Int -> Int -> (Expr -> Expr) -> [Expr]
mkNExprsWithPrefix map_size expr_size f = map (\e -> iterate f e !! expr_size) (mkNExprs map_size expr_size)


{- *********************************************************************
*                                                                      *
               Benchmarks
*                                                                      *
********************************************************************* -}

criterionDiagSizes, criterionAllSizes :: [Int]
-- criterionDiagSizes = [100, 1000]
criterionDiagSizes = [10, 100, 1000]
-- criterionDiagSizes = [10, 100, 1000, 10000]

criterionAllSizes = [10, 100, 1000, 10000]

criterion :: [Benchmark]
criterion =
  [ bgroup "filler, so that the first line begins with a leading comma" []
  , rnd_lookup
  , rnd_lookup_app1
  , rnd_lookup_app2
  , rnd_lookup_lam
  , rnd_lookup_one
  , rnd_insert_lookup_one
  , rnd_fromList_app1
  , rnd_fromList
  , rnd_union
  , rnd_union_app1
  , rnd_fold
  ]
  where
    with_map_of_exprs :: forall em. MapAPI em => Int -> Int -> ([Expr] -> em -> Benchmark) -> Benchmark
    with_map_of_exprs e m k =
      env (pure (mkNExprs e m)) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: em)) $ \(expr_map :: em) ->
      k exprs expr_map

    bench_all_variants :: String -> [Int] -> [Int] -> (forall m. MapAPI m => m -> Int -> Int -> Benchmark) -> Benchmark
    bench_all_variants name map_sizes expr_sizes f =
      bgroup name $
      flip map map_sizes $ \n ->
      bgroup (show n) $
      flip map expr_sizes $ \m ->
      bgroup (show m)
        [ bgroup "ExprMap" [f (emptyMap :: ExprMap Int)      n m]
        , bgroup "Map"     [f (emptyMap :: Map Expr Int)     n m]
        , bgroup "HashMap" [f (emptyMap :: HashMap Expr Int) n m]
        ]
    {-# INLINE bench_all_variants #-}

    bench_diag_variants :: String -> [Int] -> (forall m. MapAPI m => m -> Int -> Int -> Benchmark) -> Benchmark
    bench_diag_variants name sizes f =
      bgroup name $
      flip map sizes $ \n ->
      bgroup (show n)
        [ bgroup "ExprMap" [f (emptyMap :: ExprMap Int)      n n]
        , bgroup "Map"     [f (emptyMap :: Map Expr Int)     n n]
        , bgroup "HashMap" [f (emptyMap :: HashMap Expr Int) n n]
        ]
    {-# INLINE bench_diag_variants #-}

    rnd_lookup :: Benchmark
    rnd_lookup = bench_all_variants "lookup" criterionAllSizes criterionAllSizes $ \(_ :: em) n m ->
      with_map_of_exprs @em n m $ \exprs expr_map ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_one :: Benchmark
    rnd_lookup_one = bench_diag_variants "lookup_one" criterionDiagSizes $ \(_ :: em) n m ->
      with_map_of_exprs @em n m $ \exprs expr_map ->
      bench "" $ nf (`lookupMap` expr_map) (head exprs) -- exprs is random, so head is as good as any

    rnd_insert_lookup_one :: Benchmark
    rnd_insert_lookup_one = bench_all_variants "insert_lookup_one" criterionAllSizes criterionAllSizes $ \(_ :: em) n m ->
      with_map_of_exprs @em n m $ \_exprs expr_map ->
      env (pure (runGenDet m genClosedExpr)) $ \e ->
      bench "" $ nf (\e' -> lookupMap e' (insertMap e' (n+1) expr_map)) e

    rnd_lookup_lam :: Benchmark
    rnd_lookup_lam = bench_diag_variants "lookup_lam" criterionDiagSizes $ \(_ :: em) n m ->
      env (pure (mkNExprsWithPrefix n m (Lam "$"))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: em)) $ \(expr_map :: em) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_app1 :: Benchmark
    rnd_lookup_app1 = bench_all_variants "lookup_app1" criterionAllSizes criterionAllSizes $ \(_ :: em) n m ->
      env (pure (mkNExprsWithPrefix n m (Var "$" `App`))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: em)) $ \(expr_map :: em) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_lookup_app2 :: Benchmark
    rnd_lookup_app2 = bench_diag_variants "lookup_app2" criterionDiagSizes $ \(_ :: em) n m ->
      env (pure (mkNExprsWithPrefix n m (`App` Var "$"))) $ \exprs ->
      env (pure ((mapFromList $ zip exprs [0..]) :: em)) $ \(expr_map :: em) ->
      bench "" $ nf (map (`lookupMap` expr_map)) exprs

    rnd_fromList :: Benchmark
    rnd_fromList = bench_all_variants "fromList" criterionAllSizes criterionAllSizes $ \(_ :: em) n m ->
      env (pure (flip zip [0..] $ mkNExprs n m)) $ \pairs ->
      bench "" $ nf (mapFromList :: [(Expr, Int)] -> em) pairs

    rnd_fromList_app1 :: Benchmark
    rnd_fromList_app1 = bench_diag_variants "fromList_app1" criterionDiagSizes $ \(_ :: em) n m ->
      env (pure (flip zip [0..] $ mkNExprsWithPrefix n m (Var "$" `App`))) $ \pairs ->
      bench "" $ nf (mapFromList :: [(Expr, Int)] -> em) pairs

    rnd_union :: Benchmark
    rnd_union = bench_all_variants "union" criterionAllSizes criterionAllSizes $ \(_ :: em) n m ->
      with_map_of_exprs @em n m $ \_exprs expr_map1 ->
      with_map_of_exprs @em (n+1) (m+1) $ \_exprs expr_map2 ->
      bench "" $ nf (uncurry unionMap :: (em, em) -> em) (expr_map1, expr_map2)

    rnd_union_app1 :: Benchmark
    rnd_union_app1 = bench_diag_variants "union_app1" criterionDiagSizes $ \(_ :: em) n m ->
      env (pure (mkNExprsWithPrefix n m (Var "$" `App`))) $ \exprs1 ->
      env (pure ((mapFromList $ zip exprs1 [0..]) :: em)) $ \(expr_map1 :: em) ->
      env (pure (mkNExprsWithPrefix (n+1) (m+1) (Var "$" `App`))) $ \exprs2 ->
      env (pure ((mapFromList $ zip exprs2 [0..]) :: em)) $ \(expr_map2 :: em) ->
      bench "" $ nf (uncurry unionMap :: (em, em) -> em) (expr_map1, expr_map2)

    rnd_fold :: Benchmark
    rnd_fold = bench_diag_variants "fold" criterionDiagSizes $ \(_ :: em) n m ->
      with_map_of_exprs @em n m $ \_exprs expr_map ->
      bench "" $ whnf (\em -> fold (+) em 0) expr_map

weighSizes :: [Int]
weighSizes = [10, 100, 1000, 1000, 10000]

-- Called weigh, because I started out by using the Weigh benchmark framework,
-- but it delivered incorrect results. Still like the name
weigh :: IO ()
weigh = do
  space
  space_lam
  space_app1
  space_app2
  where
    weigh_all_variants :: String -> [Int] -> [Int] -> (forall m. MapAPI m => String -> m -> Int -> Int -> IO ()) -> IO ()
    weigh_all_variants pref map_sizes expr_sizes f =
      flip mapM_ map_sizes $ \n -> flip mapM_ expr_sizes $ \m -> do
        f (concat [pref, "/", show n, "/", show m, "/ExprMap"]) (emptyMap :: ExprMap Int)      n m
        f (concat [pref, "/", show n, "/", show m, "/Map"])     (emptyMap :: Map Expr Int)     n m
        f (concat [pref, "/", show n, "/", show m, "/HashMap"]) (emptyMap :: HashMap Expr Int) n m

    space :: IO ()
    space = weigh_all_variants "space" weighSizes weighSizes $ \pref (_ :: em) n m -> do
      let map = mapFromList (flip zip [0..] $ mkNExprs n m) :: em
      s <- recursiveSize $!! map
      putStrLn (pref ++ ": " ++ show s)

    space_lam :: IO ()
    space_lam = weigh_all_variants "space_lam" weighSizes weighSizes $ \pref (_ :: em) n m -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n m (Lam "$")) :: em
      s <- recursiveSize $!! map
      putStrLn (pref ++ ": " ++ show s)

    space_app1 :: IO ()
    space_app1 = weigh_all_variants "space_app1" weighSizes weighSizes $ \pref (_ :: em) n m -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n m (Var "$" `App`)) :: em
      s <- recursiveSize $!! map
      putStrLn (pref ++ ": " ++ show s)

    space_app2 :: IO ()
    space_app2 = weigh_all_variants "space_app2" weighSizes weighSizes $ \pref (_ :: em) n m -> do
      let map = mapFromList (flip zip [0..] $ mkNExprsWithPrefix n m (`App` Var "$")) :: em
      s <- recursiveSize $!! map
      putStrLn (pref ++ ": " ++ show s)

m :: MapAPI m => Int -> m
m n = mapFromList $ zip (runGenDet n $ vectorOf n genClosedExpr) [0..]


