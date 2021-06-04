module Arbitrary (Env, genEnv, boundVars, genOpenType, genClosedType, printSample) where

import GenTrieMap

import qualified Data.Set as Set
import Data.Char
import Text.Show
import Control.Arrow
import Control.Monad.Trans.Reader
import Control.Monad.Trans.Class

import qualified Test.QuickCheck as QC

newtype Env = Env { nextFree :: Int }

instance Show Env where
  showsPrec _ = showListWith showString . boundVars

idx2TyVar :: Int -> TyVar
idx2TyVar n | n <= 26   = [chr (ord 'a' + n)]
            | otherwise = "t" ++ show n

boundVars :: Env -> [TyVar]
boundVars (Env n) = map idx2TyVar [0..n-1]

mkEnvWithNVars :: Int -> Env
mkEnvWithNVars = Env

emptyEnv :: Env
emptyEnv = mkEnvWithNVars 0

genEnv :: QC.Gen Env
genEnv = QC.sized $ \size -> QC.elements (map mkEnvWithNVars [0..size])

genClosedType :: QC.Gen Type
genClosedType = genOpenType emptyEnv

genOpenType :: Env -> QC.Gen Type
genOpenType env = QC.sized $ \size ->
  QC.frequency $ concat
    [ [ (1, genTyConTy env)     ]
    , [ (2, genTyVarTy env)     | not $ null $ boundVars env ]
    , [ (size, genFunTy env)    ]
    , [ (size, genForAllTy env) ]
    ]

genTyConTy, genTyVarTy, genFunTy, genForAllTy :: Env -> QC.Gen Type
genTyConTy  _   = QC.elements (map TyConTy ["Int", "Bool", "Char", "Void", "String"])
genTyVarTy  env = QC.elements (map TyVarTy (boundVars env))
genFunTy    env = FunTy <$> QC.scale (subtract 1) (genOpenType env)
                        <*> QC.scale (subtract 1) (genOpenType env)
genForAllTy env = withBoundTyVar env $ \tv env' ->
  ForAllTy tv <$> QC.scale (subtract 1) (genOpenType env')

withBoundTyVar :: Env -> (TyVar -> Env -> QC.Gen a) -> QC.Gen a
withBoundTyVar env f = QC.oneof $ fresh : [ shadowing | not $ null $ boundVars env ]
  where
    fresh = do
      let tv   = idx2TyVar (nextFree env)
          env' = env { nextFree = nextFree env + 1 }
      f tv env'
    shadowing = do
      tv <- QC.elements (boundVars env)
      f tv env

-- Just for prototyping
printSample = QC.sample genClosedType
