{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE StrictData #-}

{-# OPTIONS_GHC -Wincomplete-patterns -Wunused-binds -W unused-imports #-}

-- | This module presents
--
--   * The generic 'TrieMap' type class, featuring a broad number of
--     trie-generic functions
--   * The generic trie map transformers 'SEMap' and 'ListMap'
--   * A trie map 'ExprMap' that is modelled after @ExprLMap@
--   * The matching 'MTrieMap' type class and its instances 'MExprMap'
--     and 'MSEMap'
--   * A nice wrapper type 'PatMap' that makes use of 'MExprMap'
--   * A demonstration of how 'alterTM' can be further generalised to yield
--     an implementation of the lensy 'at' combinator in a new sub-class
--     'TrieMapLens'.
--
module TrieMap where

import qualified Data.List as List
import Data.List( foldl' )
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.IntMap (IntMap)
import qualified Data.IntMap as IntMap
import Data.Kind
import Control.Applicative
import Control.Monad
import Control.Monad.Trans.State
import Data.Maybe
import Text.PrettyPrint as PP hiding (empty)
import Debug.Trace
import Data.Char
import qualified Text.Read as Read
import qualified Text.ParserCombinators.ReadP as ReadP
import Data.Tree

import Lens.Micro
import Lens.Micro.Internal
import Lens.Micro.GHC () -- At instance for containers
import Data.Functor.Identity

{- *********************************************************************
*                                                                      *
                   Expr
*                                                                      *
********************************************************************* -}

type Var    = String

data Expr = Var Var
          | App Expr Expr
          | Lam Var Expr

anyFreeVarsOfExpr :: (Var -> Bool) -> Expr -> Bool
-- True if 'p' returns True of any free variable
-- of the expr; False otherwise
anyFreeVarsOfExpr p e
  = go Set.empty e
  where
    go bvs (Var v) | v `Set.member` bvs = False
                   | otherwise          = p v
    go bvs (App e1 e2) = go bvs e1 || go bvs e2
    go bvs (Lam v e)   = go (Set.insert v bvs) e

{- *********************************************************************
*                                                                      *
               DeBruijn numbers
*                                                                      *
********************************************************************* -}

type DBNum = Int
data DeBruijnEnv = DBE { dbe_next :: !DBNum
                       , dbe_env  :: !(Map Var DBNum) }

emptyDBE :: DeBruijnEnv
emptyDBE = DBE { dbe_next = 0, dbe_env = Map.empty }

extendDBE :: Var -> DeBruijnEnv -> DeBruijnEnv
extendDBE tv (DBE { dbe_next = bv, dbe_env = env })
  = DBE { dbe_next = bv+1, dbe_env = Map.insert tv bv env }

lookupDBE :: Var -> DeBruijnEnv -> Maybe DBNum
lookupDBE v (DBE { dbe_env = env }) = Map.lookup v env


{- *********************************************************************
*                                                                      *
               Bound variables
*                                                                      *
********************************************************************* -}

type BoundVar    = DBNum   -- Bound variables are deBruijn leveled
type BoundVarEnv = DeBruijnEnv
type BoundVarMap = IntMap

emptyBVM :: BoundVarMap a
emptyBVM = IntMap.empty

lookupBoundVarOcc :: BoundVar -> BoundVarMap a -> Maybe a
lookupBoundVarOcc = IntMap.lookup

foldrBVM :: (v -> a -> a) -> BoundVarMap v -> a -> a
foldrBVM k m z = foldr k z m

alterBoundVarOcc :: BoundVar -> XT a -> BoundVarMap a -> BoundVarMap a
alterBoundVarOcc tv xt = IntMap.alter xt tv

-- | @ModAlpha a@ represents @a@ modulo alpha-renaming.  This is achieved
-- by equipping the value with a 'DeBruijnEnv', which tracks an on-the-fly deBruijn
-- numbering.  This allows us to define an 'Eq' instance for @ModAlpha a@, even
-- if this was not (easily) possible for @a@.
data ModAlpha a = A !DeBruijnEnv !a
  deriving Functor

-- | Synthesizes a @ModAlpha a@ from an @a@, by assuming that there are no
-- bound binders (an empty 'DeBruijnEnv').  This is usually what you want if there
-- isn't already a 'DeBruijnEnv' in scope.
deBruijnize :: a -> ModAlpha a
deBruijnize = A emptyDBE

instance Show e => Show (ModAlpha e) where
  show (A _ e) = show e

noCaptured :: DeBruijnEnv -> Expr -> Bool
-- True iff no free var of the type is bound by DeBruijnEnv
noCaptured dbe e
  = not (anyFreeVarsOfExpr captured e)
  where
    captured v = isJust (lookupDBE v dbe)

type AlphaExpr = ModAlpha Expr

eqAlphaExpr :: AlphaExpr -> AlphaExpr -> Bool
eqAlphaExpr ae1@(A _ (App s1 t1)) ae2@(A _ (App s2 t2))
  = eqAlphaExpr (s1 <$ ae1) (s2 <$ ae2) &&
    eqAlphaExpr (t1 <$ ae1) (t2 <$ ae2)

eqAlphaExpr (A env1 (Var v1)) (A env2 (Var v2))
  = case (lookupDBE v1 env1, lookupDBE v2 env2) of
      (Just bv1, Just bv2) -> bv1 == bv2
      (Nothing,   Nothing) -> v1 == v2
      _                    -> False

eqAlphaExpr (A env1 (Lam v1 e1)) (A env2 (Lam v2 e2))
  = eqAlphaExpr (A (extendDBE v1 env1) e1)
             (A (extendDBE v2 env2) e2)

eqAlphaExpr _ _ = False

instance Eq AlphaExpr where -- for the TrieMap instance (e.g., lookupSEM)
  (==) = eqAlphaExpr

eqClosedExpr :: Expr -> Expr -> Bool
eqClosedExpr a b = eqAlphaExpr (deBruijnize a) (deBruijnize b)

--
-- Ord Expr instance for benchmarks:
--

exprTag :: Expr -> Int
exprTag Var{} = 0
exprTag App{} = 1
exprTag Lam{} = 2
{-# INLINE exprTag #-}

cmpAlphaExpr :: AlphaExpr -> AlphaExpr -> Ordering
cmpAlphaExpr (A env1 (App f1 a1)) (A env2 (App f2 a2))
  = cmpAlphaExpr (A env1 f1) (A env2 f2) Prelude.<> cmpAlphaExpr (A env1 a1) (A env2 a2)

cmpAlphaExpr (A env1 (Lam v1 e1)) (A env2 (Lam v2 e2))
  = cmpAlphaExpr (A (extendDBE v1 env1) e1) (A (extendDBE v2 env2) e2)

cmpAlphaExpr (A env1 (Var v1))    (A env2 (Var v2))
  = case (lookupDBE v1 env1, lookupDBE v2 env2) of
      (Just bvi1, Just bvi2) -> compare bvi1 bvi2
      (Nothing,   Nothing)   -> compare v1 v2
      (Just _,    Nothing)   -> GT
      (Nothing,   Just _)    -> LT
cmpAlphaExpr (A _ e1) (A _ e2)
  = compare (exprTag e1) (exprTag e2)

instance Eq Expr where
  a == b = eqAlphaExpr (deBruijnize a) (deBruijnize b)
instance Ord Expr where
  compare a b = cmpAlphaExpr (deBruijnize a) (deBruijnize b)

{- *********************************************************************
*                                                                      *
                   Free variables
*                                                                      *
********************************************************************* -}

type FreeVar = Var
type FreeVarMap a = Map FreeVar a

emptyFVM :: FreeVarMap a
emptyFVM = Map.empty

foldrFVM :: (v -> a -> a) -> FreeVarMap v -> a -> a
foldrFVM k m z = foldr k z m

lookupFreeVarOcc :: FreeVar -> FreeVarMap a -> Maybe a
lookupFreeVarOcc = Map.lookup

alterFreeVarOcc :: FreeVar -> XT a -> FreeVarMap a -> FreeVarMap a
alterFreeVarOcc v xt = Map.alter xt v

{- *********************************************************************
*                                                                      *
                   Pattern variables
*                                                                      *
********************************************************************* -}

type PatVar    = DBNum
type PatVarEnv = DeBruijnEnv
type PatVarMap = IntMap
type PatSubst  = PatVarMap AlphaExpr -- Maps PatVar :-> AlphaExpr

emptyPVM :: PatVarMap a
emptyPVM = IntMap.empty

lookupPatSubst :: PatVar -> PatSubst -> AlphaExpr
lookupPatSubst key subst
  = case IntMap.lookup key subst of
      Just e  -> e
      Nothing -> error ("lookupPatSubst " ++ show key)

alterPatVarOcc :: PatVar -> XT a -> PatVarMap a -> PatVarMap a
alterPatVarOcc tv xt tm = IntMap.alter xt tv tm

foldrPVM :: (v -> a -> a) -> PatVarMap v -> a -> a
foldrPVM f m a = foldr f a m

--
-- Canonicalisation of pattern variables:
--

data Occ
  = Free  !FreeVar
  | Bound !BoundVar
  | Pat   !PatVar
  deriving Eq

canonOcc :: PatVarEnv -> BoundVarEnv -> Var -> Occ
canonOcc pe be v
  | Just bv <- lookupDBE v be = Bound bv
  | Just pv <- lookupDBE v pe = Pat pv
  | otherwise                 = Free v
{-# INLINE canonOcc #-}

canonPatEnv :: Set Var -> Expr -> PatVarEnv
canonPatEnv pvars  = go emptyDBE emptyDBE
  where
    go penv benv e = case e of
      Var v
        | Free _ <- canonOcc penv benv v -- not already present in penv
        , v `Set.member` pvars
        -> extendDBE v penv
        | otherwise
        -> penv
      App f a -> go (go penv benv f) benv a
      Lam b e -> go penv (extendDBE b benv) e

{- *********************************************************************
*                                                                      *
                  The TrieMap class
*                                                                      *
********************************************************************* -}

class Eq (TrieKey tm) => TrieMap tm where
   type TrieKey tm :: Type
   emptyTM        :: tm v
   lookupTM       :: TrieKey tm -> tm v -> Maybe v
   alterTM        :: TrieKey tm -> XT v -> tm v -> tm v
   unionWithTM    :: (v -> v -> v) -> tm v -> tm v -> tm v
   foldrTM        :: (v -> a -> a) -> tm v -> a -> a
   fromListWithTM :: ([r] -> v) -> [(TrieKey tm, r)] -> tm v

--   mapTM    :: (a->b) -> tm a -> tm b
--   filterTM :: (a -> Bool) -> tm a -> tm a
--   unionTM  ::  tm a -> tm a -> tm a

insertTM :: TrieMap tm => TrieKey tm -> v -> tm v -> tm v
insertTM k v = alterTM k (const $ Just v)

type XT v = Maybe v -> Maybe v  -- How to alter a non-existent elt (Nothing)
                                --               or an existing elt (Just)

-- Recall that
--   Control.Monad.(>=>) :: (a -> Maybe b) -> (b -> Maybe c) -> a -> Maybe c

(|>) :: a -> (a->b) -> b     -- Reverse application
infixl 0 |>
x |> f = f x

(|>>) :: TrieMap m2 => (XT (m2 a) -> m1 (m2 a) -> m1 (m2 a))
                    -> (m2 a -> m2 a)
                    -> m1 (m2 a) -> m1 (m2 a)
infixl 1 |>>
(|>>) f g = f (Just . g . deMaybe)

deMaybe :: TrieMap m => Maybe (m a) -> m a
deMaybe Nothing  = emptyTM
deMaybe (Just m) = m

(|>>>) :: MTrieMap m2 => (XT (m2 a) -> m1 (m2 a) -> m1 (m2 a))
                      -> (m2 a -> m2 a)
                      -> m1 (m2 a) -> m1 (m2 a)
infixl 1 |>>>
(|>>>) f g = f (Just . g . deMaybeMTM)

deMaybeMTM :: MTrieMap m => Maybe (m a) -> m a
deMaybeMTM Nothing  = emptyMTM
deMaybeMTM (Just m) = m

foldrMaybe :: (v -> a -> a) -> Maybe v -> a -> a
foldrMaybe _ Nothing  z = z
foldrMaybe f (Just v) z = f v z

{- *********************************************************************
*                                                                      *
                  Singleton and empty map
*                                                                      *
********************************************************************* -}

data SEMap tm v
  = EmptySEM
  | SingleSEM (TrieKey tm) v
  | MultiSEM  (tm v)

deriving instance (Show v, Show (tm v), Show (TrieKey tm))
               => Show (SEMap tm v)


instance TrieMap tm => TrieMap (SEMap tm) where
  type TrieKey (SEMap tm) = TrieKey tm
  emptyTM     = EmptySEM
  lookupTM    = lookupSEM
  alterTM     = alterSEM
  unionWithTM = unionWithSEM
  foldrTM     = foldrSEM
  fromListWithTM = fromListWithSEM

lookupSEM :: TrieMap tm => TrieKey tm -> SEMap tm v -> Maybe v
lookupSEM !_  EmptySEM = Nothing
lookupSEM tk (SingleSEM pk v) | tk == pk  = Just v
                              | otherwise = Nothing
lookupSEM tk (MultiSEM tm) = lookupTM tk tm


alterSEM :: TrieMap tm => TrieKey tm -> XT v -> SEMap tm v -> SEMap tm v
alterSEM k xt EmptySEM
  = case xt Nothing of
      Nothing -> EmptySEM
      Just v  -> SingleSEM k v
alterSEM k1 xt tm@(SingleSEM k2 v2)
  | k1 == k2 = case xt (Just v2) of
                  Nothing -> EmptySEM
                  Just v' -> SingleSEM k2 v'
  | otherwise = case xt Nothing of
                  Nothing -> tm
                  Just v1  -> MultiSEM $ alterTM k1 (\_ -> Just v1)
                                       $ alterTM k2 (\_ -> Just v2)
                                       $ emptyTM
alterSEM k xt (MultiSEM tm)
  = MultiSEM (alterTM k xt tm)

unionWithSEM :: TrieMap tm => (v -> v -> v) -> SEMap tm v -> SEMap tm v -> SEMap tm v
unionWithSEM _ EmptySEM        m        = m
unionWithSEM _ m               EmptySEM = m
unionWithSEM f (SingleSEM k1 v1) (SingleSEM k2 v2)
  | k1 == k2 = SingleSEM k1 (f v1 v2)
  | otherwise = MultiSEM $ alterTM k1 (\_ -> Just v1)
                         $ alterTM k2 (\_ -> Just v2)
                         $ emptyTM
unionWithSEM _ (MultiSEM tm)   (SingleSEM k v)
  = MultiSEM $ alterTM k (\_ -> Just v) tm
unionWithSEM _ (SingleSEM k v) (MultiSEM tm)
  = MultiSEM $ alterTM k xt tm
  where
    xt Nothing = Just v
    xt old     = old
unionWithSEM f (MultiSEM tm1)  (MultiSEM tm2)
  = MultiSEM $ unionWithTM f tm1 tm2

foldrSEM :: TrieMap tm => (v -> a -> a) -> SEMap tm v -> a -> a
foldrSEM _ EmptySEM        z = z
foldrSEM f (SingleSEM _ v) z = f v z
foldrSEM f (MultiSEM tm)   z = foldrTM f tm z

fromListWithSEM :: TrieMap tm => ([r] -> v) -> [(TrieKey tm, r)] -> SEMap tm v
fromListWithSEM _ [] = EmptySEM
fromListWithSEM f [(k,v)] = SingleSEM k (f [v])
fromListWithSEM f kvs = MultiSEM $ fromListWithTM f kvs

{- *********************************************************************
*                                                                      *
                  ListMap
*                                                                      *
********************************************************************* -}

type ListMap tm = SEMap (ListMap' tm)

data ListMap' tm a
  = LM { lm_nil  :: Maybe a
       , lm_cons :: tm (ListMap tm a) }

instance TrieMap tm => TrieMap (ListMap' tm) where
   type TrieKey (ListMap' tm) = [TrieKey tm]
   emptyTM        = emptyLM
   lookupTM       = lookupLM
   alterTM        = alterLM
   unionWithTM    = unionWithLM
   foldrTM        = foldrLM
   fromListWithTM = fromListWithLM

emptyLM :: TrieMap tm => ListMap' tm a
emptyLM = LM { lm_nil = Nothing, lm_cons = emptyTM }

lookupLM :: TrieMap tm => [TrieKey tm] -> ListMap' tm v -> Maybe v
lookupLM key (LM {..})
  = case key of
      []     -> lm_nil
      (k:ks) -> lm_cons |> lookupTM k >=> lookupTM ks

alterLM :: TrieMap tm => [TrieKey tm] -> XT v -> ListMap' tm v -> ListMap' tm v
alterLM ks xt tm@(LM {..})
  = case ks of
      []      -> tm { lm_nil  = lm_nil |> xt }
      (k:ks') -> tm { lm_cons = lm_cons |> alterTM k |>> alterTM ks' xt }

unionWithMaybe :: (v -> v -> v) -> Maybe v -> Maybe v -> Maybe v
unionWithMaybe f (Just v1) (Just v2) = Just (f v1 v2)
unionWithMaybe _ m1        m2        = m1 `mplus` m2

unionWithLM :: TrieMap tm => (v -> v -> v) -> ListMap' tm v -> ListMap' tm v -> ListMap' tm v
unionWithLM f m1 m2
  = LM { lm_nil = unionWithMaybe f (lm_nil m1) (lm_nil m2)
       , lm_cons = unionWithTM (unionWithTM f) (lm_cons m1) (lm_cons m2) }

foldrLM :: TrieMap tm => (v -> a -> a) -> ListMap' tm v -> a -> a
foldrLM f (LM {..}) = foldrMaybe f lm_nil . foldrTM (foldrTM f) lm_cons

partitionLists :: [([k], v)] -> ([v], [(k,[k],v)])
partitionLists kvs = go kvs ([],[])
  where
    go []             (!nils, !conss) = (nils, conss)
    go (([],v):kvs)   (nils, conss) = go kvs (v:nils, conss)
    go ((k:ks,v):kvs) (nils, conss) = go kvs (nils,   (k,ks,v):conss)

fromListWithLM :: TrieMap tm => ([r] -> v) -> [([TrieKey tm], r)] -> ListMap' tm v
fromListWithLM f kvs
  = LM { lm_nil  = if Prelude.null nils then Nothing else Just (f nils)
       , lm_cons = fromListWithTM (fromListWithTM f) [ (k, (ks, v)) | (k, ks, v) <- conss ] }
  where
    (nils, conss) = partitionLists kvs

{- *********************************************************************
*                                                                      *
                  ExprMap
*                                                                      *
********************************************************************* -}

type ExprMap = SEMap ExprMap'
data ExprMap' a
  = EM { em_bvar :: BoundVarMap a    -- Occurrence of a lambda-bound var
       , em_fvar :: FreeVarMap a     -- Occurrence of a free var
       , em_app  :: ExprMap (ExprMap a)
       , em_lam  :: ExprMap a
       }

deriving instance (Show (TrieKey ExprMap'), Show v)
               => Show (ExprMap' v)

instance TrieMap ExprMap' where
  type TrieKey ExprMap' = AlphaExpr
  emptyTM        = mkEmptyExprMap
  lookupTM       = lookupEM
  alterTM        = alterEM
  unionWithTM    = unionWithEM
  foldrTM        = foldrEM
  fromListWithTM = fromListWithEM

emptyExprMap :: ExprMap a
emptyExprMap = EmptySEM

mkEmptyExprMap :: ExprMap' a
mkEmptyExprMap
  = EM { em_fvar = emptyFVM
       , em_bvar = emptyBVM
       , em_app  = emptyExprMap
       , em_lam  = emptyExprMap }

lookupEM :: AlphaExpr -> ExprMap' v -> Maybe v
lookupEM ae@(A benv e) (EM { .. }) = case e of
  Var x     -> case lookupDBE x benv of
    Just bv -> em_bvar |> lookupBoundVarOcc bv
    Nothing -> em_fvar |> lookupFreeVarOcc  x
  App e1 e2 -> em_app  |>  lookupTM (e1 <$ ae)
                       >=> lookupTM (e2 <$ ae)
  Lam x e   -> em_lam  |> lookupTM (A (extendDBE x benv) e)

alterEM :: AlphaExpr -> XT v -> ExprMap' v -> ExprMap' v
alterEM ae@(A benv e) xt m@(EM {..}) = case e of
  Var x -> case lookupDBE x benv of
    Just bv -> m { em_bvar = alterBoundVarOcc bv xt em_bvar }
    Nothing -> m { em_fvar = alterFreeVarOcc  x  xt em_fvar }
  App e1 e2 -> m { em_app = em_app |> alterTM (e1 <$ ae) |>> alterTM (e2 <$ ae) xt }
  Lam x e   -> m { em_lam = em_lam |> alterTM (A (extendDBE x benv) e) xt }

unionWithEM :: (v -> v -> v) -> ExprMap' v -> ExprMap' v -> ExprMap' v
unionWithEM f m1 m2
  = EM { em_bvar = IntMap.unionWith f (em_bvar m1) (em_bvar m2)
       , em_fvar = Map.unionWith f (em_fvar m1) (em_fvar m2)
       , em_app  = unionWithTM (unionWithTM f) (em_app m1) (em_app m2)
       , em_lam  = unionWithTM f (em_lam m1) (em_lam m2) }

foldrEM :: forall a v. (v -> a -> a) -> ExprMap' v -> a -> a
foldrEM f (EM {..}) z
  = let !z1 = foldrTM f em_lam z in
    let !z2 = foldrTM (\em z -> z `seq` foldrTM f em z) em_app z1 in
    let !z3 = foldrFVM f em_fvar z2 in
    foldrBVM f em_bvar z3

partitionExprs :: [(AlphaExpr, v)] -> ([(FreeVar, v)], [(BoundVar, v)], [(ModAlpha (Expr, Expr), v)], [(AlphaExpr,v)])
partitionExprs = foldr go ([], [], [], [])
  where
    go (ae@(A benv e), val) (!fvars, !bvars, !apps, !lams) = case e of
      Var v | Just bv <- lookupDBE v benv -> (fvars, (bv,val):bvars, apps, lams)
            | otherwise                   -> ((v,val):fvars, bvars, apps, lams)
      App f a -> (fvars, bvars, ((f,a) <$ ae, val):apps, lams)
      Lam b e -> (fvars, bvars, apps, (A (extendDBE b benv) e,val):lams)

fromListWithEM :: ([r] -> v) -> [(AlphaExpr, r)] -> ExprMap' v
fromListWithEM f kvs
  = EM { em_bvar = IntMap.map f $ foldr (\(bv,val) -> IntMap.alter (cons_bucket val) bv) IntMap.empty bvars
       , em_fvar = Map.map f $ foldr (\(v,val) -> Map.alter (cons_bucket val) v) Map.empty fvars
       , em_app  = fromListWithTM (fromListWithTM f) [ (A benv f, (A benv a, v)) | (A benv (f, a), v) <- apps]
       , em_lam  = fromListWithTM f lams
       }
  where
    (fvars, bvars, apps, lams)  = partitionExprs kvs
    cons_bucket val Nothing     = Just [val]
    cons_bucket val (Just vals) = Just (val:vals)


-- | For debugging purposes. Draw with 'containers:Data.Tree.drawTree' or
-- 'tree-view:Data.Tree.View.showTree'. The latter uses much less screen space.
-- Random example output (with 'Data.Tree.showTree'):
--
-- > .
-- > └╴@
-- >    ├╴λ
-- >    │  └╴sing(Q)
-- >    │     └╴sing(K)
-- >    │        └╴0
-- >    └╴@
-- >       └╴sing(W)
-- >          └╴sing(F)
-- >             └╴sing(λa. a)
-- >                └╴1
--
-- Compare that to showing it:
--
-- > MultiSEM (EM {em_bvar = fromList [], em_fvar = fromList [], em_app = MultiSEM (EM {em_bvar = fromList [], em_fvar = fromList [], em_app = SingleSEM W (SingleSEM F (SingleSEM λa. a 1)), em_lit = fromList [], em_lam = SingleSEM Q (SingleSEM K 0)}), em_lit = fromList [], em_lam = EmptySEM})
--
exprMapToTree :: Show v => ExprMap v -> Tree String
exprMapToTree = Node "." . go_sem (\v -> [ Node (show v) [] ])
  where
    mk_node_nonempty _   []       = []
    mk_node_nonempty lbl children = [ Node lbl children ]

    go_sem _      EmptySEM = []
    go_sem go_val (SingleSEM k v) = mk_node_nonempty ("sing(" ++ show k ++ ")") (go_val v)
    go_sem go_val (MultiSEM em) = go_em go_val em

    go_em go_val EM{..} = concat
      [ go_fvar go_val em_fvar
      , go_bvar go_val em_bvar
      , go_lam  go_val em_lam
      , go_app  go_val em_app
      ]

    go_bvar go_val bvm = [ Node ("bvar(" ++ show k ++ ")") (go_val v) | (k,v) <- IntMap.toList bvm ]
    go_fvar go_val m   = [ Node ("fvar(" ++ k      ++ ")") (go_val v) | (k,v) <- Map.toList m ]
    go_lam  go_val em  = mk_node_nonempty "λ" (go_sem go_val em)
    go_app :: (v -> Forest String) -> ExprMap (ExprMap v) -> Forest String
    go_app  go_val em  = mk_node_nonempty "@" (go_sem (go_sem go_val) em)

{- *********************************************************************
*                                                                      *
                  Pat
*                                                                      *
********************************************************************* -}

-- | '(<$)' through two functors
(<$$) :: (Functor f, Functor g) => b -> f (g a) -> f (g b)
b <$$ fga = (b <$) <$> fga

data Pat a = P !PatVarEnv !a
  deriving Functor

instance Show e => Show (Pat e) where
  show (P _ e) = show e


{- *********************************************************************
*                                                                      *
                  Matchable type class
*                                                                      *
********************************************************************* -}

newtype MatchState e = MS (PatVarMap e)
  deriving Show

emptyMS :: MatchState e
emptyMS = MS emptyPVM

getMatchingSubst :: MatchState e -> PatVarMap e
getMatchingSubst (MS subst) = subst

hasMatch :: PatVar -> MatchState e -> Maybe e
hasMatch pv (MS subst) = IntMap.lookup pv subst

addMatch :: PatVar -> e -> MatchState e -> MatchState e
addMatch pv e (MS ms) = MS (IntMap.insert pv e ms)

class Eq (Pat e) => Matchable e where
  equate :: PatVar -> e -> MatchState e -> Maybe (MatchState e)
  match  :: Pat e -> e -> MatchState e -> Maybe (MatchState e)

instance Eq (Pat (ModAlpha Expr)) where
  (==) = eqPatExpr

eqPatExpr :: Pat (ModAlpha Expr) -> Pat (ModAlpha Expr) -> Bool
eqPatExpr v1@(P _ (A _ (App s1 t1))) v2@(P _ (A _ (App s2 t2)))
  = eqPatExpr (s1 <$$ v1) (s2 <$$ v2) &&
    eqPatExpr (t1 <$$ v1) (t2 <$$ v2)

eqPatExpr (P penv1 (A benv1 (Var v1))) (P penv2 (A benv2 (Var v2)))
  = canonOcc penv1 benv1 v1 == canonOcc penv2 benv2 v2

eqPatExpr (P penv1 (A benv1 (Lam v1 e1))) (P penv2 (A benv2 (Lam v2 e2)))
  = eqPatExpr (P penv1 (A (extendDBE v1 benv1) e1))
            (P penv2 (A (extendDBE v2 benv2) e2))

eqPatExpr _ _ = False

instance Matchable (ModAlpha Expr) where
  equate = equateExpr
  match  = matchExpr

equateExpr :: PatVar -> ModAlpha Expr -> MatchState (ModAlpha Expr) -> Maybe (MatchState (ModAlpha Expr))
equateExpr pv (A benv e) ms = case hasMatch pv ms of
  Just (A _ sol)
    | eqClosedExpr e sol  -> Just ms
    | otherwise           -> Nothing
  Nothing
    | noCaptured benv e   -> Just (addMatch pv (A emptyDBE e) ms)
    | otherwise           -> Nothing

traceWith f x = trace (f x) x

matchExpr :: Pat AlphaExpr -> AlphaExpr -> MatchState AlphaExpr -> Maybe (MatchState AlphaExpr)
matchExpr pat@(P penv (A benv_pat e_pat)) tar@(A benv_tar e_tar) ms =
  -- traceWith (\res -> show ms ++ "  ->  matchExpr " ++ show pat ++ "   " ++ show tar ++ "  -> " ++ show res) $
  case (e_pat, e_tar) of
  (Var v, _) -> case canonOcc penv benv_pat v of
    Pat pv -> equate pv tar ms
    occ -> do
      Var v2 <- pure e_tar
      guard (occ == canonOcc emptyDBE benv_tar v2)
      pure ms
  (App f1 a1, App f2 a2) -> match (f1 <$$ pat) (f2 <$ tar) ms >>= match (a1 <$$ pat) (a2 <$ tar)
  (Lam b1 e1, Lam b2 e2) -> match (P penv (A (extendDBE b1 benv_pat) e1)) (A (extendDBE b2 benv_tar) e2) ms
  (_, _) -> Nothing

{- *********************************************************************
*                                                                      *
                  MatchResult monad
*                                                                      *
********************************************************************* -}

newtype MatchResult e a = MR (StateT (MatchState e) [] a)
  deriving (Functor, Applicative, Monad, Alternative, MonadPlus)

liftMaybe :: Maybe a -> MatchResult e a
liftMaybe ma = MR $ StateT $ \ms -> case ma of
  Just a  -> [(a, ms)]
  Nothing -> []

refine :: (MatchState e -> Maybe (MatchState e)) -> MatchResult e ()
refine f = MR $ StateT $ \ms -> case f ms of
  Just ms' -> [((), ms')]
  Nothing  -> []

runMatchResult :: MatchResult e a -> [(PatVarMap e, a)]
runMatchResult (MR f)
  = [ (getMatchingSubst ms, a) | (a, ms) <- runStateT f emptyMS ]

{- *********************************************************************
*                                                                      *
                  Matching TrieMap class
*                                                                      *
********************************************************************* -}

class Matchable (Term tm) => MTrieMap tm where
  type Term tm :: Type
  emptyMTM :: tm a
  lookupPatMTM :: Term tm -> tm a -> MatchResult (Term tm) a
  alterPatMTM  :: Pat (Term tm) -> XT a -> tm a -> tm a

{- *********************************************************************
*                                                                      *
                  Matching singleton and empty maps
*                                                                      *
********************************************************************* -}

data MSEMap tm v
  = EmptyMSEM
  | SingleMSEM (Pat (Term tm)) v
  | MultiMSEM  (tm v)

deriving instance (Show v, Show (tm v), Show (Term tm))
               => Show (MSEMap tm v)

instance MTrieMap tm => MTrieMap (MSEMap tm) where
  type Term (MSEMap tm) = Term tm
  emptyMTM     = EmptyMSEM
  lookupPatMTM = lookupPatMSEM
  alterPatMTM  = alterPatMSEM

lookupPatMSEM
  :: MTrieMap tm => Term tm -> MSEMap tm a -> MatchResult (Term tm) a
lookupPatMSEM k m = case m of
  EmptyMSEM -> empty
  SingleMSEM pat v -> do
    refine (match pat k)
    pure v
  MultiMSEM m -> lookupPatMTM k m

alterPatMSEM
  :: MTrieMap tm
  => Pat (Term tm) -> XT a -> MSEMap tm a -> MSEMap tm a
alterPatMSEM k xt EmptyMSEM
  = case xt Nothing of
      Nothing -> EmptyMSEM
      Just v  -> SingleMSEM k v
alterPatMSEM k1 xt tm@(SingleMSEM k2 v2)
  | k1 == k2 = case xt (Just v2) of
                  Nothing -> EmptyMSEM
                  Just v' -> SingleMSEM k2 v'
  | otherwise = case xt Nothing of
                  Nothing -> tm
                  Just v1  -> MultiMSEM $ alterPatMTM k1 (\_ -> Just v1)
                                        $ alterPatMTM k2 (\_ -> Just v2)
                                        $ emptyMTM
alterPatMSEM k xt (MultiMSEM tm)
  = MultiMSEM (alterPatMTM k xt tm)


{- *********************************************************************
*                                                                      *
                  Matching ExprMap
*                                                                      *
********************************************************************* -}

type MExprMap = MSEMap MExprMap'
data MExprMap' a
  = MEM { mem_pvar   :: PatVarMap a      -- Occurrence of a pattern var
        , mem_bvar   :: BoundVarMap a    -- Occurrence of a lam-bound var
        , mem_fvar   :: FreeVarMap a     -- Occurrence of a completely free var
        , mem_app    :: MExprMap (MExprMap a)
        , mem_lam    :: MExprMap a
        }

deriving instance Show v => Show (MExprMap' v)

instance MTrieMap MExprMap' where
  type Term MExprMap' = AlphaExpr
  emptyMTM     = mkEmptyMExprMap
  lookupPatMTM = lookupPatMM
  alterPatMTM  = alterPatMM

mkEmptyMExprMap :: MExprMap' a
mkEmptyMExprMap
  = MEM { mem_pvar = emptyPVM
        , mem_fvar = emptyFVM
        , mem_bvar = emptyBVM
        , mem_app  = emptyMTM
        , mem_lam  = emptyMTM }

lookupPatMM :: AlphaExpr -> MExprMap' a -> MatchResult AlphaExpr a
lookupPatMM ae@(A benv e) (MEM { .. })
  = match <|> decompose
  where
    match = mem_pvar |> IntMap.toList |> map match_one |> msum
    match_one (pv, x) = refine (equate pv ae) >> pure x

    decompose = case e of
      Var x     -> case lookupDBE x benv of
        Just bv -> mem_bvar |> liftMaybe . lookupBoundVarOcc bv
        Nothing -> mem_fvar |> liftMaybe . lookupFreeVarOcc  x
      App e1 e2 -> mem_app  |>  lookupPatMTM (e1 <$ ae)
                            >=> lookupPatMTM (e2 <$ ae)
      Lam x e   -> mem_lam  |> lookupPatMTM (A (extendDBE x benv) e)

alterPatMM :: Pat AlphaExpr -> XT a -> MExprMap' a -> MExprMap' a
alterPatMM pat@(P penv (A benv e)) xt m@(MEM {..}) = case e of
  Var x -> case canonOcc penv benv x of
    Pat pv   -> m { mem_pvar = alterPatVarOcc   pv xt mem_pvar }
    Bound bv -> m { mem_bvar = alterBoundVarOcc bv xt mem_bvar }
    Free fv  -> m { mem_fvar = alterFreeVarOcc  fv xt mem_fvar }
  App e1 e2  -> m { mem_app  = mem_app |> alterPatMTM (e1 <$$ pat) |>>> alterPatMTM (e2 <$$ pat) xt }
  Lam x e    -> m { mem_lam  = mem_lam |> alterPatMTM (P penv (A (extendDBE x benv) e)) xt }

-- An ad-hoc definition of foldrMM, because I don't want to define another
-- (duplicate) type class method. We need this one in the testuite to extract
-- all patterns from a PatSet

foldrMM :: (v -> a -> a) -> a -> MExprMap v -> a
foldrMM f z m = sem f z m
  where
    sem :: (v -> a -> a) -> a -> MSEMap MExprMap' v -> a
    sem _ z EmptyMSEM = z
    sem f z (SingleMSEM _ v) = f v z
    sem f z (MultiMSEM m) = expr f z m
    expr :: (v -> a -> a) -> a -> MExprMap' v -> a
    expr f z MEM{..} =
      let !z1 = sem f z mem_lam in
      let !z2 = sem (\mem z -> z `seq` sem f z mem) z1 mem_app in
      let !z3 = foldrFVM f mem_fvar z2 in
      let !z4 = foldrBVM f mem_bvar z3 in
      let !z5 = foldrPVM f mem_pvar z4 in
      z5


{- *********************************************************************
*                                                                      *
                  PatMap
*                                                                      *
********************************************************************* -}

type Match a = ([(Var, PatVar)], a)
type PatMap a = MExprMap (Match a)
type PatSet = PatMap Expr

emptyPatMap :: PatMap a
emptyPatMap = emptyMTM

insertPM :: forall a. [Var]   -- Pattern variables
                      -> Expr -- Pattern
                      -> a -> PatMap a -> PatMap a
insertPM pvars e x pm = alterPatMTM pat xt pm
  where
    penv = canonPatEnv (Set.fromList pvars) e
    pat = P penv (deBruijnize e)
    xt :: XT (Match a)
    xt _ = Just (map inst_key pvars, x)
     -- The "_" means just overwrite previous value
     where
        inst_key :: Var -> (Var, PatVar)
        inst_key v = case lookupDBE v penv of
                         Nothing  -> error ("Unbound PatVar " ++ v)
                         Just pv -> (v, pv)

matchPM :: Expr -> PatMap a -> [ ([(Var,Expr)], a) ]
matchPM e pm
  = [ (map (lookup subst) prs, x)
    | (subst, (prs, x)) <- runMatchResult $ lookupPatMTM (deBruijnize e) pm ]
  where
    lookup :: PatSubst -> (Var, PatVar) -> (Var, Expr)
    lookup subst (v, pv) = (v, e)
      where A _ e = lookupPatSubst pv subst

deletePM :: forall a. [Var]   -- Pattern variables
                      -> Expr -- Pattern
                      -> PatMap a -> PatMap a
deletePM pvars e pm = alterPatMTM pat (const Nothing) pm
  where
    penv = canonPatEnv (Set.fromList pvars) e
    pat = P penv (deBruijnize e)

mkPatMap :: [([Var], Expr, a)] -> PatMap a
mkPatMap = foldr (\(tmpl_vs, e, a) -> insertPM tmpl_vs e a) emptyPatMap

mkPatSet :: [([Var], Expr)] -> PatSet
mkPatSet = mkPatMap . map (\(tmpl_vs, e) -> (tmpl_vs, e, e))

elemsPatSet :: PatMap Expr -> [([Var], Expr)]
elemsPatSet = foldrMM (\(pks, e) pats -> (map fst pks, e):pats) []

-- | See also 'exprMapToTree'
patMapToTree :: Show v => PatMap v -> Tree String
patMapToTree = Node "." . go_sem (\v -> [ Node (show v) [] ])
  where
    mk_node_nonempty _   []       = []
    mk_node_nonempty lbl children = [ Node lbl children ]

    go_sem _      EmptyMSEM = []
    go_sem go_val (SingleMSEM k v) = mk_node_nonempty ("sing(" ++ show k ++ ")") (go_val v)
    go_sem go_val (MultiMSEM mem) = go_mem go_val mem

    go_mem go_val MEM{..} = concat
      [ go_fvar go_val mem_fvar
      , go_bvar go_val mem_bvar
      , go_pvar go_val mem_pvar
      , go_lam  go_val mem_lam
      , go_app  go_val mem_app
      ]

    go_pvar go_val pvm = [ Node ("pvar(" ++ show k ++ ")") (go_val v) | (k,v) <- IntMap.toList pvm ]
    go_bvar go_val bvm = [ Node ("bvar(" ++ show k ++ ")") (go_val v) | (k,v) <- IntMap.toList bvm ]
    go_fvar go_val m   = [ Node ("fvar(" ++ k      ++ ")") (go_val v) | (k,v) <- Map.toList m ]
    go_lam  go_val em  = mk_node_nonempty "λ" (go_sem go_val em)
    go_app :: (v -> Forest String) -> MExprMap (MExprMap v) -> Forest String
    go_app  go_val mem  = mk_node_nonempty "@" (go_sem (go_sem go_val) mem)


{- *********************************************************************
*                                                                      *
          Lensy versions, basically instances of Lens.Micro.At
*                                                                      *
********************************************************************* -}

-- | We would like to make TrieMap an instance of the 'At' type class
-- from microlens, but I don't see how to define the necessary type family
-- instances..
class TrieMap tm => TrieMapLens tm where
  atTM  :: TrieKey tm -> Lens' (tm v) (Maybe v)
  nullTM :: tm v -> Bool

alterTM' :: TrieMapLens tm => TrieKey tm -> XT v -> tm v -> tm v
alterTM' k xt m = runIdentity $ atTM k (Identity . xt) m

lookupTM' :: TrieMapLens tm => TrieKey tm -> tm v -> Maybe v
lookupTM' k m = getConst $ atTM k Const m

instance (TrieMapLens tm, TrieKey tm ~ k) => TrieMapLens (SEMap tm) where
  atTM = atSEM
  nullTM = nullSEM

nullSEM :: TrieMapLens tm => SEMap tm v -> Bool
nullSEM EmptySEM      = True
-- nullSEM (MultiSEM tm) = nullTM tm -- Invariant: MultiSEM is never empty
nullSEM _             = False

atSEM :: TrieMapLens tm => TrieKey tm -> Lens' (SEMap tm v) (Maybe v)
atSEM !k xt EmptySEM
  = xt Nothing <&> \case
      Nothing -> EmptySEM
      Just v  -> SingleSEM k v
atSEM k1 xt tm@(SingleSEM k2 v2)
  | k1 == k2 = xt (Just v2) <&> \case
                  Nothing -> EmptySEM
                  Just v' -> SingleSEM k2 v'
  | otherwise = xt Nothing <&> \case
                  Nothing -> tm
                  Just v1  -> MultiSEM $ insertTM k1 v1 (insertTM k2 v2 emptyTM)
atSEM k xt (MultiSEM tm)
  = atTM k xt tm <&> \tm' -> if nullTM tm' then EmptySEM else MultiSEM tm'

instance TrieMapLens tm => TrieMapLens (ListMap' tm) where
  atTM = atList
  nullTM = nullList

lens_lm_nil :: Lens' (ListMap' tm v) (Maybe v)
lens_lm_nil xt lm@(LM { .. }) = xt lm_nil <&> \nil' -> lm { lm_nil = nil' }

lens_lm_cons :: Lens' (ListMap' tm v) (tm (ListMap tm v))
lens_lm_cons xt lm@(LM { .. }) = xt lm_cons <&> \cons' -> lm { lm_cons = cons' }

nullList :: TrieMapLens tm => ListMap' tm v -> Bool
nullList (LM {..}) = isNothing lm_nil && nullTM lm_cons

-- | Like the 'non' combinator from microlens, but specialised to
-- 'emptyTM'/'nullTM' instead of 'Eq'.
nonEmpty :: TrieMapLens tm => Lens' (Maybe (tm v)) (tm v)
nonEmpty afb s = f <$> afb (fromMaybe emptyTM s)
  where f y = if nullTM y then Nothing else Just y
{-# INLINE nonEmpty #-}

atList :: TrieMapLens tm => [TrieKey tm] -> Lens' (ListMap' tm v) (Maybe v)
atList []     = lens_lm_nil
atList (k:ks) = lens_lm_cons . atTM k . nonEmpty . atTM ks

atBoundVarOcc :: BoundVar -> Lens' (BoundVarMap v) (Maybe v)
atBoundVarOcc tv xt tm = IntMap.alterF xt tv tm

atFreeVarOcc :: Var -> Lens' (FreeVarMap v) (Maybe v)
atFreeVarOcc tv xt tm = Map.alterF xt tv tm

instance TrieMapLens ExprMap' where
  atTM = atExpr
  nullTM = nullExpr

lens_em_bvar :: Lens' (ExprMap' a) (BoundVarMap a)
lens_em_bvar xf em@(EM { .. }) = xf em_bvar <&> \bvar' -> em { em_bvar = bvar' }

lens_em_fvar :: Lens' (ExprMap' a) (FreeVarMap a)
lens_em_fvar xf em@(EM { .. }) = xf em_fvar <&> \fvar' -> em { em_fvar = fvar' }

lens_em_app :: Lens' (ExprMap' a) (ExprMap (ExprMap a))
lens_em_app xf em@(EM { .. }) = xf em_app <&> \app' -> em { em_app = app' }

lens_em_lam :: Lens' (ExprMap' a) (ExprMap a)
lens_em_lam xf em@(EM { .. }) = xf em_lam <&> \lam' -> em { em_lam = lam' }

nullExpr :: ExprMap' v -> Bool
nullExpr (EM {..}) =  Prelude.null em_fvar && Prelude.null em_bvar
                   && nullTM em_app && nullTM em_lam

atExpr :: AlphaExpr -> Lens' (ExprMap' v) (Maybe v)
atExpr (A dbe e) = case e of
  Var x     -> case lookupDBE x dbe of
                  Just bv -> lens_em_bvar . at bv -- NB: at from microlens's `At (IntMap v)` instance
                  Nothing -> lens_em_fvar . at x
  App e1 e2 -> lens_em_app . atTM (A dbe e1) . nonEmpty . atTM (A dbe e2)
  Lam x e   -> lens_em_lam . atTM (A (extendDBE x dbe) e)

-- We would have to define these microlens instances for every TrieMapLens. It's
-- weirdly redundant, I'm only doing so for ExprMap' here:
type instance Index (ExprMap' v) = AlphaExpr
type instance IxValue (ExprMap' v) = v
instance Ixed (ExprMap' v) where
  ix = ixAt -- defined in microlens
instance At (ExprMap' v) where
  at = atTM

{- *********************************************************************
*                                                                      *
                   Pretty-printing
*                                                                      *
********************************************************************* -}

appPrec, lamPrec :: Read.Prec
lamPrec = Read.minPrec
appPrec = lamPrec+1

-- | Example output: @F (λa. G) (H I) (λb. J b)@
instance Show Expr where
  showsPrec _ (Var v)      = showString v
  showsPrec p (App f arg)  = showParen (p > appPrec) $
    showsPrec appPrec f . showString " " . showsPrec (appPrec+1) arg
  showsPrec p (Lam b body) = showParen (p > lamPrec) $
    showString "λ" . showString b . showString ". " . showsPrec lamPrec body

-- | The default 'ReadP.many1' function leads to ambiguity. What a terrible API.
greedyMany, greedyMany1 :: ReadP.ReadP a -> ReadP.ReadP [a]
greedyMany p  = greedyMany1 p ReadP.<++ pure []
greedyMany1 p = (:) <$> p <*> greedyMany p

-- | This monster parses Exprs in the REPL etc. It parses names that start
-- with an upper-case letter as literals and lower-case names as variables.
--
-- Accepts syntax like
-- @F (λa. G) (H I) (λb. J b)@
--
-- >>> read "F (λa. G) (H I) (λb. J b)" :: Expr
-- F (λa. G) (H I) (λb. J b)
instance Read Expr where
  readPrec = Read.parens $ Read.choice
    [ do
        Read.Ident v <- Read.lexP
        guard (all isAlphaNum v)
        pure $ Var v
    , Read.prec appPrec $ do
        -- Urgh. Just ignore the code here as long as it works
        let spaces1 = greedyMany1 (ReadP.satisfy isSpace)
        (f:args) <- Read.readP_to_Prec $ \prec ->
          ReadP.sepBy1 (Read.readPrec_to_P Read.readPrec (prec+1)) spaces1
        guard $ not $ List.null args
        pure (foldl' App f args)
    , Read.prec lamPrec $ do
        c <- Read.get
        guard (c `elem` "λΛ@#%\\") -- multiple short-hands for Lam
        Var v <- Read.readPrec
        '.' <- Read.get
        Lam v <$> Read.readPrec
    ]

class Pretty a where
  ppr :: a -> Doc

pprTrace :: String -> Doc -> a -> a
pprTrace s d x = trace (show (text s <+> d)) x

pprTraceWith s f a = pprTrace s (f a) a

commaSep :: [Doc] -> Doc
commaSep ds = fsep (go ds)
  where
    go []     = []
    go [d]    = [d]
    go (d:ds) = (d PP.<> comma) : go ds

instance Pretty Int where
   ppr i = int i

instance Pretty Char where
   ppr c = char c

instance {-# OVERLAPPING #-} Pretty String where
   ppr s = doubleQuotes (text s)

instance (Pretty a, Pretty b) => Pretty (a,b) where
  ppr (x,y) = parens (ppr x PP.<> comma <+> ppr y)

instance Pretty a => Pretty [a] where
  ppr xs = brackets (commaSep (map ppr xs))

instance (Pretty k, Pretty a) => Pretty (Map k a) where
   ppr m = brackets $ commaSep [ ppr k <+> text ":->" <+> ppr v
                               | (k,v) <- Map.toList m ]

instance Pretty a => Pretty (IntMap a) where
   ppr m = brackets $ vcat [ ppr k <+> text ":->" <+> ppr v
                           | (k,v) <- IntMap.toList m ]

instance Pretty a => Pretty (Maybe a) where
  ppr Nothing  = text "Nothing"
  ppr (Just x) = text "Just" <+> ppr x

instance Pretty Expr where
  ppr e = text (show e)

instance Pretty DeBruijnEnv where
  ppr (DBE { .. }) = text "DBE" PP.<>
                     braces (sep [ text "dbe_next =" <+> ppr dbe_next
                                 , text "dbe_env =" <+> ppr dbe_env ])

instance (Pretty a) => Pretty (ModAlpha a) where
  ppr (A bv a) = text "A" PP.<> braces (sep [ppr bv, ppr a])

instance (Pretty a) => Pretty (Pat a) where
  ppr (P pv a) = text "P" PP.<> braces (sep [ppr pv, ppr a])

instance (Pretty v, Pretty (tm v), Pretty (TrieKey tm))
      => Pretty (SEMap tm v) where
  ppr EmptySEM        = text "EmptySEM"
  ppr (SingleSEM k v) = text "SingleSEM" <+> ppr k <+> ppr v
  ppr (MultiSEM tm)   = ppr tm

instance Pretty a => Pretty (ExprMap' a) where
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "em_bvar =" <+> ppr em_bvar
                    , text "em_fvar =" <+> ppr em_fvar
                    , text "em_app ="  <+> ppr em_app
                    , text "em_lam ="  <+> ppr em_lam ])

instance (Pretty v, Pretty (tm v), Pretty (Pat (Term tm)))
      => Pretty (MSEMap tm v) where
  ppr EmptyMSEM        = text "EmptyMSEM"
  ppr (SingleMSEM k v) = text "SingleMSEM" <+> ppr k <+> ppr v
  ppr (MultiMSEM tm)   = ppr tm

instance Pretty a => Pretty (MExprMap' a) where
  ppr (MEM {..}) = text "MEM" <+> braces (vcat
                    [ text "mem_pvar =" <+> ppr mem_pvar
                    , text "mem_bvar =" <+> ppr mem_bvar
                    , text "mem_fvar =" <+> ppr mem_fvar
                    , text "mem_app ="  <+> ppr mem_app
                    , text "mem_lam ="  <+> ppr mem_lam ])
