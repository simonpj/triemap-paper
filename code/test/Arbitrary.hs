module Arbitrary where

import GenTrieMap

import qualified Data.Set as Set
import qualified Data.Tree.View
import Data.Char
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
    [ [ (1,                genLit env) ]
    , [ (2,                genVar env) | not $ null $ boundVars env ]
    , [ (size * 8 `div` 7, genApp env) ]
    , [ (size `div` 4,     genLam env) ]
    ]

-- | This defn leads to good correlation between QC size and expr sizes
appFactor :: Int -> Int
appFactor n = n*16 `div` 31

genLit, genVar, genApp, genLam :: Env -> QC.Gen Expr
genLit _   = QC.elements (map (Lit . (:[])) ['A'..'Z'])
genVar env = QC.elements (map Var (boundVars env))
genApp env = App <$> QC.scale appFactor (genOpenExpr env)
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

isqrt :: Int -> Int
isqrt = floor . sqrt . fromIntegral

mkExprMap :: [Expr] -> ExprMap Int
mkExprMap = foldr (\(v, k) -> insertTM (deBruijnize k) v) emptyExprMap . zip [0..]

genClosedExprMap :: QC.Gen (ExprMap Int)
genClosedExprMap = do
  sz <- QC.getSize
  traceM (show sz)
  QC.resize (isqrt sz) $ mkExprMap <$> QC.vectorOf (isqrt sz) genClosedExpr

exprDepth :: Expr -> Int
exprDepth (Lit _) = 0
exprDepth (Var _) = 0
exprDepth (App f a) = 1 + max (exprDepth f) (exprDepth a)
exprDepth (Lam _ e) = 1 + exprDepth e

exprSize :: Expr -> Int
exprSize (Lit _) = 1
exprSize (Var _) = 1
exprSize (App f a) = 1 + exprSize f + exprSize a
exprSize (Lam _ e) = 1 + exprSize e

-- Just for prototyping
printSample = QC.sample genClosedExpr
printSampleMap = do
  maps <- QC.sample' genClosedExprMap
  mapM_ (Data.Tree.View.drawTree . exprMapToTree) maps
