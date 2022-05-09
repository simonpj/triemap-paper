{-# LANGUAGE TupleSections #-}

module Arbitrary where

import TrieMap

import qualified Data.Set as Set
import qualified Data.Tree.View
import Data.Char
import Data.Maybe
import Text.Show
import Control.Arrow
import Control.Monad.Trans.Reader
import Control.Monad.Trans.Class
import Debug.Trace

import qualified Test.QuickCheck as QC

newtype Env = Env { nextFree :: Int }

instance Show Env where
  showsPrec _ = showListWith showString . boundVars

idx2Var :: Int -> Var
idx2Var n | n <= 26   = [chr (ord 'a' + n)]
          | otherwise = "t" ++ show n

boundVars :: Env -> [Var]
boundVars (Env n) = map idx2Var [0..n-1]

mkEnvWithNVars :: Int -> Env
mkEnvWithNVars = Env

emptyEnv :: Env
emptyEnv = mkEnvWithNVars 0

genEnv :: QC.Gen Env
genEnv = QC.sized $ \size -> QC.elements (map mkEnvWithNVars [0..size])

genClosedExpr :: QC.Gen Expr
genClosedExpr = genOpenExpr emptyEnv

genOpenExpr :: Env -> QC.Gen Expr
genOpenExpr env = QC.sized $ \size ->
  QC.frequency $ concat
    [ [ (1,                genFreeVar env)  ]
    , [ (2,                genBoundVar env) | not $ null $ boundVars env ]
    , [ (size * 8 `div` 7, genApp env)      ]
    , [ (size `div` 4,     genLam env)      ]
    ]

-- | This defn leads to good correlation between QC size and expr sizes
appFactor :: Int -> Int
appFactor n = n*16 `div` 31

genFreeVar, genBoundVar, genApp, genLam :: Env -> QC.Gen Expr
genFreeVar  env = QC.elements (map (Var . (:[])) ['A'..'Z']) -- upper case is never a bound var
genBoundVar env = QC.elements (map Var (boundVars env))
genApp      env = App <$> QC.scale appFactor (genOpenExpr env)
                 <*> QC.scale appFactor (genOpenExpr env)
genLam env = withBoundVar env $ \v env' ->
  Lam v <$> QC.scale (subtract 1) (genOpenExpr env')

withBoundVar :: Env -> (Var -> Env -> QC.Gen a) -> QC.Gen a
withBoundVar env f = QC.oneof $ fresh : [ shadowing | not $ null $ boundVars env ]
  where
    fresh = do
      let tv   = idx2Var (nextFree env)
          env' = env { nextFree = nextFree env + 1 }
      f tv env'
    shadowing = do
      tv <- QC.elements (boundVars env)
      f tv env

genPattern :: QC.Gen ([Var], Expr)
genPattern = do
  e <- genEnv >>= genOpenExpr
  pure (exprFreeVars e, e)

exprFreeVars :: Expr -> [Var]
exprFreeVars e = Set.toList (go emptyDBE e)
  where
    go env (Var v) | Just _ <- lookupDBE v env = Set.empty
                   | c:_ <- v, isUpper c       = Set.empty -- uppercase chars are "constants"
                   | otherwise                 = Set.singleton v
    go env (App f a) = go env f `Set.union` go env a
    go env (Lam b e) = go (extendDBE b env) e

isqrt :: Int -> Int
isqrt = floor . sqrt . fromIntegral

mkExprMap :: [Expr] -> ExprMap Int
mkExprMap = foldr (\(v, k) -> insertTM (deBruijnize k) v) emptyExprMap . zip [0..]

genClosedExprMap :: QC.Gen (ExprMap Int)
genClosedExprMap = do
  sz <- QC.getSize
  QC.resize (isqrt sz) $ mkExprMap <$> QC.vectorOf (isqrt sz) genClosedExpr

exprDepth :: Expr -> Int
exprDepth (Var _) = 0
exprDepth (App f a) = 1 + max (exprDepth f) (exprDepth a)
exprDepth (Lam _ e) = 1 + exprDepth e

exprSize :: Expr -> Int
exprSize (Var _) = 1
exprSize (App f a) = 1 + exprSize f + exprSize a
exprSize (Lam _ e) = 1 + exprSize e

genPatMap :: QC.Gen (PatMap Int)
genPatMap = do
  sz <- QC.getSize
  pats <- QC.resize (isqrt sz) $ QC.vectorOf (isqrt sz) genPattern
  pure (mkPatMap (zipWith (\(pvs,e) i -> (pvs,e,i)) pats [0..]))

genPatSet :: QC.Gen (PatMap Expr)
genPatSet = do
  sz <- QC.getSize
  pats <- QC.resize (isqrt sz) $ QC.vectorOf (isqrt sz) genPattern
  pure (mkPatSet pats)

genInstance :: ([Var], Expr) -> QC.Gen Expr
genInstance (pvs, e) = do
  subst <- traverse (\pv -> (pv,) <$> genClosedExpr) pvs
  pure $ applySubst subst e

applySubst :: [(Var, Expr)] -> Expr -> Expr
applySubst subst e@(Var v) = fromMaybe e $ lookup v subst
applySubst subst (App arg res) = App (applySubst subst arg) (applySubst subst res)
applySubst subst (Lam v body) = Lam v (applySubst (del v subst) body)
  where
    del k' = filter (\(k,_v) -> k /= k')

-- Just for prototyping
printSample = QC.sample genClosedExpr
printSampleMap = do
  maps <- QC.sample' genClosedExprMap
  mapM_ (Data.Tree.View.drawTree . exprMapToTree) maps
