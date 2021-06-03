{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}

module Arbitrary (genType) where

import GenTrieMap

import qualified Data.Set as Set
import Data.Char
import Control.Arrow
import Control.Monad.Trans.Reader
import Control.Monad.Trans.Class

import qualified Test.QuickCheck as QC

newtype Env = Env { nextFree :: Int }

idx2TyVar :: Int -> TyVar
idx2TyVar n | n <= 26   = [chr (ord 'a' + n)]
            | otherwise = "t" ++ show n

boundVars :: Env -> [TyVar]
boundVars (Env n) = map idx2TyVar [0..n-1]

emptyEnv :: Env
emptyEnv = Env 0

newtype EnvGen a = EnvGen (ReaderT Env QC.Gen a)
  deriving newtype (Functor, Applicative, Monad)

runEnvGen :: Env -> EnvGen a -> QC.Gen a
runEnvGen env (EnvGen m) = runReaderT m env

getEnv :: EnvGen Env
getEnv = EnvGen ask

liftGen :: QC.Gen a -> EnvGen a
liftGen m = EnvGen (lift m)

getSize :: EnvGen Int
getSize = liftGen QC.getSize

scale :: (Int -> Int) -> EnvGen a -> EnvGen a
scale f g = EnvGen $ ReaderT $ \env -> QC.scale f (runEnvGen env g)

elements :: [a] -> EnvGen a
elements elts = liftGen (QC.elements elts)

frequency :: [(Int, EnvGen a)] -> EnvGen a
frequency alts = EnvGen $ ReaderT $ \env ->
  QC.frequency [ (w, runEnvGen env m) | (w, m) <- alts ]

genType :: QC.Gen Type
genType = runEnvGen emptyEnv genType'

genType' :: EnvGen Type
genType' = do
  env <- getEnv
  size <- getSize
  frequency $ concat
    [ [ (1, genTyConTy')  ]
    , [ (2, genTyVarTy')  | not $ null $ boundVars env ]
    , [ (size, genFunTy')    ]
    , [ (size, genForAllTy') ]
    ]

genTyConTy', genTyVarTy', genFunTy', genForAllTy' :: EnvGen Type
genTyConTy' = elements (map TyConTy ["Int", "Bool", "Char", "Void", "String"])
genTyVarTy' = do
  env <- getEnv
  elements (map TyVarTy (boundVars env))
genFunTy' = FunTy <$> scale (subtract 1) genType' <*> scale (subtract 1) genType'
genForAllTy' = withFreshlyBoundTyVar $ \tv ->
  ForAllTy tv <$> scale (subtract 1) genType'

withFreshlyBoundTyVar :: (TyVar -> EnvGen a) -> EnvGen a
withFreshlyBoundTyVar f = do
  env <- getEnv
  let tv   = idx2TyVar (nextFree env)
      env' = env { nextFree = nextFree env + 1 }
  liftGen $ runEnvGen env' (f tv)

-- Just for prototyping
main = QC.sample genType
