{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables,
             StandaloneDeriving, UndecidableInstances, TypeApplications,
             LambdaCase, DeriveFunctor, ViewPatterns, TupleSections,
             TypeFamilies, EmptyDataDecls #-}

{-# LANGUAGE BangPatterns, StrictData #-} -- for benchmarks

{-# OPTIONS_GHC -Wincomplete-patterns #-}

-- | This module presents
--
--   * The generic triemap transformers 'SEMap' and 'ListMap' from the paper
--   * A trie map 'ExprMap' that is modelled after @ExprLMap@ from the paper.
--   * The matching trie map type 'PatMap' from the paper.
--   * A demonstration of how 'alterTM' can be further generalised to yield
--     an implementation of the lensy 'at' combinator in a new sub-class
--     'TrieMapLens'.
--
module TrieMap where

import qualified Data.Foldable as Foldable
import qualified Data.List as List
import Data.List( foldl' )
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.IntMap (IntMap)
import qualified Data.IntMap as IntMap
import Data.Kind
import Control.Monad
import Data.Maybe
import Text.PrettyPrint as PP
import Debug.Trace
import Data.Char
import qualified Text.Read as Read
import qualified Text.ParserCombinators.ReadP as ReadP
import Data.Tree
import Bag

import Lens.Micro
import Lens.Micro.Internal
import Lens.Micro.GHC -- At instance for containers
import Data.Functor.Identity
import Data.Functor.Const

{- *********************************************************************
*                                                                      *
                   Expr
*                                                                      *
********************************************************************* -}

type Var    = String

data Occ
  = Free Var
  | Bound BoundVar
  | Pat PatVar
  deriving Eq

data Expr = Var Var
          | App Expr Expr
          | Lam Var Expr

{- Only (perhaps) needed for unification
data Expr' p = Var (OccP p)
             | App (RecP p) (RecP p)
             | Lam (BindP p) (RecP p)

type family BindP p
type family OccP p
type family RecP p

data Vanilla
type Expr = Expr' Vanilla
type instance BindP Vanilla = Var
type instance OccP  Vanilla = Var
type instance RecP  Vanilla = Expr

data CanonShallow
type CExprShallow = Expr' CanonShallow
type instance BindP CanonShallow = ()
type instance OccP  CanonShallow = Occ
type instance RecP  CanonShallow = Expr

data CanonDeep
type CExprDeep = Expr' CanonDeep
type instance BindP CanonDeep = ()
type instance OccP  CanonDeep = Occ
type instance RecP  CanonDeep = CExprDeep
-}

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

lookupBVM :: BoundVar -> BoundVarMap a -> Maybe a
lookupBVM = IntMap.lookup

extendBVM :: BoundVar -> a -> BoundVarMap a -> BoundVarMap a
extendBVM = IntMap.insert

foldBVM :: (v -> a -> a) -> BoundVarMap v -> a -> a
foldBVM k m z = foldr k z m

lkBoundVarOcc :: BoundVar -> (a, BoundVarMap v) -> Bag (a, v)
lkBoundVarOcc var (a, env) = case lookupBVM var env of
                                     Just x  -> Bag.single (a,x)
                                     Nothing -> Bag.empty

alterBoundVarOcc :: BoundVar -> XT a -> BoundVarMap a -> BoundVarMap a
alterBoundVarOcc tv xt tm = IntMap.alter xt tv tm

-- | @ModAlpha a@ represents @a@ modulo alpha-renaming.  This is achieved
-- by equipping the value with a 'DeBruijnEnv', which tracks an on-the-fly deBruijn
-- numbering.  This allows us to define an 'Eq' instance for @ModAlpha a@, even
-- if this was not (easily) possible for @a@.  Note: we purposely don't
-- export the constructor.  Make a helper function if you find yourself
-- needing it.
data ModAlpha a = A !DeBruijnEnv !a
  deriving Functor

-- | Synthesizes a @ModAlpha a@ from an @a@, by assuming that there are no
-- bound binders (an empty 'DeBruijnEnv').  This is usually what you want if there
-- isn't already a 'DeBruijnEnv' in scope.
deBruijnize :: a -> ModAlpha a
deBruijnize e = A emptyDBE e

instance Eq (ModAlpha a) => Eq (ModAlpha [a]) where
    A _   []     == A _    []       = True
    A env (x:xs) == A env' (x':xs') = A env x  == A env' x' &&
                                      A env xs == A env' xs'
    _            == _               = False

instance Eq (ModAlpha a) => Eq (ModAlpha (Maybe a)) where
    A _   Nothing  == A _    Nothing   = True
    A env (Just x) == A env' (Just x') = A env x  == A env' x'
    _              == _                = False

noCaptured :: DeBruijnEnv -> Expr -> Bool
-- True iff no free var of the type is bound by DeBruijnEnv
noCaptured dbe ty
  = not (anyFreeVarsOfExpr captured ty)
  where
    captured tv = isJust (lookupDBE tv dbe)

instance Eq (ModAlpha Expr) where
  (==) = eqDBExpr

eqDBExpr :: ModAlpha Expr -> ModAlpha Expr -> Bool
eqDBExpr (A env1 (App s1 t1)) (A env2 (App s2 t2))
  = eqDBExpr (A env1 s1) (A env2 s2) &&
    eqDBExpr (A env1 t1) (A env2 t2)

eqDBExpr (A env1 (Var v1)) (A env2 (Var v2))
  = case (lookupDBE v1 env1, lookupDBE v2 env2) of
      (Just bv1, Just bv2) -> bv1 == bv2
      (Nothing,   Nothing) -> v1 == v2
      _                    -> False

eqDBExpr a1@(A env1 (Lam v1 e1)) a2@(A env2 (Lam v2 e2))
  = eqDBExpr (A (extendDBE v1 env1) e1)
             (A (extendDBE v2 env2) e2)

eqDBExpr _ _ = False

instance Show (ModAlpha Expr) where
  show (A _ e) = show e

--
-- Ord Expr instance for benchmarks:
--

instance Eq Expr where
  a == b = deBruijnize a == deBruijnize b

exprTag :: Expr -> Int
exprTag Var{} = 0
exprTag App{} = 1
exprTag Lam{} = 2
{-# INLINE exprTag #-}

cmpDBExpr :: ModAlpha Expr -> ModAlpha Expr -> Ordering
cmpDBExpr (A env1 (App f1 a1)) (A env2 (App f2 a2))
  = cmpDBExpr (A env1 f1) (A env2 f2) Prelude.<> cmpDBExpr (A env1 a1) (A env2 a2)

cmpDBExpr (A env1 (Lam v1 e1)) (A env2 (Lam v2 e2))
  = cmpDBExpr (A (extendDBE v1 env1) e1) (A (extendDBE v2 env2) e2)

cmpDBExpr (A env1 (Var v1))    (A env2 (Var v2))
  = case (lookupDBE v1 env1, lookupDBE v2 env2) of
      (Just bvi1, Just bvi2) -> compare bvi1 bvi2
      (Nothing,   Nothing)   -> compare v1 v2
      (Just _,    Nothing)   -> GT
      (Nothing,   Just _)    -> LT
cmpDBExpr (A _ e1) (A _ e2)
  = compare (exprTag e1) (exprTag e2)

instance Ord Expr where
  compare a b = cmpDBExpr (deBruijnize a) (deBruijnize b)

{- *********************************************************************
*                                                                      *
                   Free variables
*                                                                      *
********************************************************************* -}

type FreeVar = Var
type FreeVarMap a = Map Var a

emptyFVM :: FreeVarMap a
emptyFVM = Map.empty

lookupFVM :: FreeVarMap a -> Var -> Maybe a
lookupFVM env v = Map.lookup v env

extendFVM :: FreeVarMap a -> Var -> a -> FreeVarMap a
extendFVM env v val = Map.insert v val env

foldFVM :: (v -> a -> a) -> FreeVarMap v -> a -> a
foldFVM k m z = foldr k z m

alterFreeVarOcc :: Var -> XT a -> FreeVarMap a -> FreeVarMap a
alterFreeVarOcc v xt tm = Map.alter xt v tm

lkFreeVarOcc :: Var -> (a, FreeVarMap v) -> Bag (a, v)
lkFreeVarOcc var (a, env) = case Map.lookup var env of
                                    Just x  -> Bag.single (a, x)
                                    Nothing -> Bag.empty

{- *********************************************************************
*                                                                      *
                   Pattern variables
*                                                                      *
********************************************************************* -}

type PatVar    = Int
type PatVarMap = IntMap
data PatVarEnv = PVE
  { pve_pvs :: !(Set Var)
  , pve_env :: !DeBruijnEnv }

emptyPVM :: PatVarMap a
emptyPVM = IntMap.empty

emptyPVE :: Set Var -> PatVarEnv
emptyPVE pvs = PVE{pve_pvs = pvs, pve_env = emptyDBE}

type PatSubst = PatVarMap Expr -- Maps PatVar :-> Expr

emptyPatSubst :: PatSubst
emptyPatSubst = IntMap.empty

lookupPatSubst :: PatVar -> PatSubst -> Expr
lookupPatSubst key subst
  = case IntMap.lookup key subst of
      Just ty -> ty
      Nothing -> error ("lookupPatSubst " ++ show key)

alterPatVarOcc :: PatVar -> XT a -> PatVarMap a -> PatVarMap a
alterPatVarOcc tv xt tm = IntMap.alter xt tv tm

foldPVM :: (v -> a -> a) -> PatVarMap v -> a -> a
foldPVM f m a = foldr f a m

canonOcc :: PatVarEnv -> BoundVarEnv -> Var -> (PatVarEnv, Occ)
canonOcc pe@PVE{..} be v
  | Just bv <- lookupDBE v be        = (pe,  Bound bv)
  | not (v `Set.member` pve_pvs)     = (pe,  Free v)
  | Just pv <- lookupDBE v pve_env   = (pe,  Pat pv)
  | otherwise                        = (pe', Pat pv')
  where
    pv' = dbe_next pve_env
    pe' = pe{pve_env = extendDBE v pve_env}
{-# INLINE canonOcc #-}

{- Only needed (perhaps) if we want unification
canonShallow :: PatVarEnv -> BoundVarEnv -> Expr -> (PatVarEnv, BoundVarEnv, CExprShallow)
canonShallow pe be (Lam b e) = (pe,  extendDBE b be, Lam () e)
canonShallow pe be (App f a) = (pe,  be, App f a)
canonShallow pe be (Var v)   = (pe', be, Var occ)
  where (pe', occ) = canonOcc pe be v
{-# INLINE canonShallow #-}

canonDeep :: PatVarEnv -> BoundVarEnv -> Expr -> (PatVarEnv, CExprDeep)
canonDeep pe be (Lam v e) = (pe', Lam () e')
  where (pe', e') = canonDeep pe (extendDBE v be) e
canonDeep pe be (App f a) = (pe2, App f' a')
  where (pe1, f') = canonDeep pe  be f
        (pe2, a') = canonDeep pe1 be a
canonDeep pe be (Var v)   = (pe', Var occ)
  where (pe', occ) = canonOcc pe be v
-}

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
   foldTM         :: (v -> a -> a) -> tm v -> a -> a
   -- fromListWithTM :: (v -> v -> v) -> [(TrieKey tm, v)] -> tm v

--   mapTM    :: (a->b) -> tm a -> tm b
--   filterTM :: (a -> Bool) -> tm a -> tm a
--   unionTM  ::  tm a -> tm a -> tm a

insertTM :: TrieMap tm => TrieKey tm -> v -> tm v -> tm v
insertTM k v = alterTM k (const $ Just v)

type XT v = Maybe v -> Maybe v  -- How to alter a non-existent elt (Nothing)
                                --               or an existing elt (Just)

-- Recall that
--   Control.Monad.(>=>) :: (a -> Maybe b) -> (b -> Maybe c) -> a -> Maybe c

(>.>) :: (a -> b) -> (b -> c) -> a -> c
-- Reverse function composition (do f first, then g)
infixr 1 >.>
(f >.> g) x = g (f x)


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

foldMaybe :: (v -> a -> a) -> Maybe v -> a -> a
foldMaybe f Nothing  z = z
foldMaybe f (Just v) z = f v z

{- *********************************************************************
*                                                                      *
                  Singleton and empty map
*                                                                      *
********************************************************************* -}

data SEMap k tm v
  = EmptySEM
  | SingleSEM k v
  | MultiSEM  (tm v)

deriving instance (Show v, Show (tm v), Show k)
               => Show (SEMap k tm v)


instance (TrieMap tm, TrieKey tm ~ k) => TrieMap (SEMap k tm) where
  type TrieKey (SEMap k tm) = k
  emptyTM     = EmptySEM
  lookupTM    = lookupSEM
  alterTM     = alterSEM
  unionWithTM = unionWithSEM
  foldTM      = foldSEM
  -- fromListWithTM = fromListWithSEM

lookupSEM :: TrieMap tm => TrieKey tm -> SEMap (TrieKey tm) tm v -> Maybe v
lookupSEM !_  EmptySEM = Nothing
lookupSEM tk (SingleSEM pk v) | tk == pk  = Just v
                              | otherwise = Nothing
lookupSEM tk (MultiSEM tm) = lookupTM tk tm


alterSEM :: TrieMap tm => TrieKey tm -> XT v -> SEMap (TrieKey tm) tm v -> SEMap (TrieKey tm) tm v
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

unionWithSEM :: TrieMap tm => (v -> v -> v) -> SEMap (TrieKey tm) tm v -> SEMap (TrieKey tm) tm v -> SEMap (TrieKey tm) tm v
unionWithSEM _ EmptySEM        m        = m
unionWithSEM _ m               EmptySEM = m
unionWithSEM f (SingleSEM k1 v1) (SingleSEM k2 v2)
  | k1 == k2 = SingleSEM k1 (f v1 v2)
  | otherwise = MultiSEM $ alterTM k1 (\_ -> Just v1)
                         $ alterTM k2 (\_ -> Just v2)
                         $ emptyTM
unionWithSEM f (MultiSEM tm)   (SingleSEM k v)
  = MultiSEM $ alterTM k (\_ -> Just v) tm
unionWithSEM f (SingleSEM k v) (MultiSEM tm)
  = MultiSEM $ alterTM k xt tm
  where
    xt Nothing = Just v
    xt old     = old
unionWithSEM f (MultiSEM tm1)  (MultiSEM tm2)
  = MultiSEM $ unionWithTM f tm1 tm2

foldSEM :: TrieMap tm => (v -> a -> a) -> SEMap (TrieKey tm) tm v -> a -> a
foldSEM _ EmptySEM        z = z
foldSEM f (SingleSEM _ v) z = f v z
foldSEM f (MultiSEM tm)   z = foldTM f tm z

-- fromListWithSEM :: TrieMap tm => (v -> v -> v) -> [(TrieKey tm, v)] -> SEMap tm v
-- fromListWithSEM _ [] = EmptySEM
-- fromListWithSEM _ [(k,v)] = SingleSEM k v
-- fromListWithSEM f kvs = MultiSEM $ fromListWithTM f kvs

{- *********************************************************************
*                                                                      *
                  ListMap
*                                                                      *
********************************************************************* -}

type ListMap tm = SEMap [TrieKey tm] (ListMap' tm)

data ListMap' tm a
  = LM { lm_nil  :: Maybe a
       , lm_cons :: tm (ListMap tm a) }

instance TrieMap tm => TrieMap (ListMap' tm) where
   type TrieKey (ListMap' tm) = [TrieKey tm]
   emptyTM     = emptyList
   lookupTM    = lookupList
   alterTM     = alterList
   unionWithTM = unionWithList
   foldTM      = foldList
   -- fromListWithTM = undefined

emptyList :: TrieMap tm => ListMap' tm a
emptyList = LM { lm_nil = Nothing, lm_cons = emptyTM }

lookupList :: TrieMap tm => [TrieKey tm] -> ListMap' tm v -> Maybe v
lookupList key (LM {..})
  = case key of
      []     -> lm_nil
      (k:ks) -> lm_cons |> lookupTM k >=> lookupTM ks

alterList :: TrieMap tm => [TrieKey tm] -> XT v -> ListMap' tm v -> ListMap' tm v
alterList ks xt tm@(LM {..})
  = case ks of
      []      -> tm { lm_nil  = lm_nil |> xt }
      (k:ks') -> tm { lm_cons = lm_cons |> alterTM k |>> alterTM ks' xt }

unionWithMaybe :: (v -> v -> v) -> Maybe v -> Maybe v -> Maybe v
unionWithMaybe f (Just v1) (Just v2) = Just (f v1 v2)
unionWithMaybe _ m1        m2        = m1 `mplus` m2

unionWithList :: TrieMap tm => (v -> v -> v) -> ListMap' tm v -> ListMap' tm v -> ListMap' tm v
unionWithList f m1 m2
  = LM { lm_nil = unionWithMaybe f (lm_nil m1) (lm_nil m2)
       , lm_cons = unionWithTM (unionWithTM f) (lm_cons m1) (lm_cons m2) }

foldList :: TrieMap tm => (v -> a -> a) -> ListMap' tm v -> a -> a
foldList f (LM {..}) = foldMaybe f lm_nil . foldTM (foldTM f) lm_cons


{- *********************************************************************
*                                                                      *
                  ExprMap
*                                                                      *
********************************************************************* -}


type ExprMap = SEMap (ModAlpha Expr) ExprMap'

data ExprMap' a
  = EM { em_bvar :: BoundVarMap a    -- Occurrence of a forall-bound tyvar
       , em_fvar :: FreeVarMap a     -- Occurrence of a completely free tyvar

       , em_app  :: ExprMap (ExprMap a)
       , em_lam  :: ExprMap a
       }

deriving instance (Show (TrieKey ExprMap'), Show v)
               => Show (ExprMap' v)

instance TrieMap ExprMap' where
  type TrieKey ExprMap' = ModAlpha Expr
  emptyTM     = mkEmptyExprMap
  lookupTM    = lookupExpr
  alterTM     = alterExpr
  unionWithTM = unionWithExpr
  foldTM      = foldExpr
  -- fromListWithTM = fromListWithExpr

emptyExprMap :: ExprMap a
emptyExprMap = EmptySEM

mkEmptyExprMap :: ExprMap' a
mkEmptyExprMap
  = EM { em_fvar = emptyFVM
       , em_bvar = emptyBVM
       , em_app  = emptyExprMap
       , em_lam  = emptyExprMap }

lookupExpr :: ModAlpha Expr -> ExprMap' v -> Maybe v
lookupExpr (A dbe e) (EM { .. })
  = case e of
      Var x     -> case lookupDBE x dbe of
                     Just bv -> em_bvar |> lookupBVM bv
                     Nothing -> em_fvar |> Map.lookup x
      App e1 e2 -> em_app |>  lookupTM (A dbe e1)
                          >=> lookupTM (A dbe e2)
      Lam x e   -> em_lam |> lookupTM (A (extendDBE x dbe) e)

alterExpr :: ModAlpha Expr -> XT v -> ExprMap' v -> ExprMap' v
alterExpr (A dbe e) xt m@(EM {..})
  = case e of
      Var x -> case lookupDBE x dbe of
                  Just bv -> m { em_bvar = alterBoundVarOcc bv xt em_bvar }
                  Nothing -> m { em_fvar = alterFreeVarOcc  x  xt em_fvar }

      App e1 e2 -> m { em_app = em_app |> alterTM (A dbe e1) |>> alterTM (A dbe e2) xt }
      Lam x e   -> m { em_lam = em_lam |> alterTM (A (extendDBE x dbe) e) xt }

unionWithExpr :: (v -> v -> v) -> ExprMap' v -> ExprMap' v -> ExprMap' v
unionWithExpr f m1 m2
  = EM { em_bvar = IntMap.unionWith f (em_bvar m1) (em_bvar m2)
       , em_fvar = Map.unionWith f (em_fvar m1) (em_fvar m2)
       , em_app  = unionWithTM (unionWithTM f) (em_app m1) (em_app m2)
       , em_lam  = unionWithTM f (em_lam m1) (em_lam m2) }

foldExpr :: forall a v. (v -> a -> a) -> ExprMap' v -> a -> a
foldExpr f (EM {..}) z
  = let !z1 = foldTM f em_lam z in
    let !z2 = foldTM (\em z -> z `seq` foldTM f em z) em_app z1 in
    let !z3 = foldFVM f em_fvar z2 in
    foldBVM f em_bvar z3


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
    mk_node_nonempty lbl []       = []
    mk_node_nonempty lbl children = [ Node lbl children ]

    go_sem go_val EmptySEM = []
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
    go_lit  go_val m   = [ Node ("lit("  ++ k      ++ ")") (go_val v) | (k,v) <- Map.toList m ]
    go_lam  go_val em  = mk_node_nonempty "λ" (go_sem go_val em)
    go_app :: (v -> Forest String) -> ExprMap (ExprMap v) -> Forest String
    go_app  go_val em  = mk_node_nonempty "@" (go_sem (go_sem go_val) em)

{- *********************************************************************
*                                                                      *
                  V
*                                                                      *
********************************************************************* -}

-- | Like 'ModAlpha', but also includes a 'PatVarEnv'.
data V a = V !PatVarEnv !BoundVarEnv !a
  deriving Functor

eqVFoldable :: (Eq (V a), Foldable f) => V (f a) -> V (f a) -> Bool
eqVFoldable va@(V _ _ as) vb@(V _ _ bs) =
  map (<$ va) (Foldable.toList as) == map (<$ vb) (Foldable.toList bs)

instance Eq (V a) => Eq (V [a]) where
  (==) = eqVFoldable

instance Eq (V a) => Eq (V (Maybe a)) where
  (==) = eqVFoldable

instance Eq (V Expr) where
  (==) = eqVExpr

eqVExpr :: V Expr -> V Expr -> Bool
eqVExpr v1@(V _ _ (App s1 t1)) v2@(V _ _ (App s2 t2))
  = eqVExpr (s1 <$ v1) (s2 <$ v2) &&
    eqVExpr (t1 <$ v1) (t2 <$ v2)

eqVExpr (V penv1 benv1 (Var v1)) (V penv2 benv2 (Var v2))
  = snd (canonOcc penv1 benv1 v1) == snd (canonOcc penv2 benv2 v2)

eqVExpr (V penv1 benv1 (Lam v1 e1)) (V penv2 benv2 (Lam v2 e2))
  = eqVExpr (V penv1 (extendDBE v1 benv1) e1)
            (V penv2 (extendDBE v2 benv2) e2)

eqVExpr _ _ = False

instance Show (V Expr) where
  show (V _ _ e) = show e

{- *********************************************************************
*                                                                      *
                  Matching ExprMap
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

class Matchable e where
  equate :: PatVar -> ModAlpha e -> MatchState e -> Maybe (MatchState e)
  match  :: ModAlpha e -> ModAlpha e -> (PatVarEnv, MatchState e) -> Maybe (PatVarEnv, MatchState e)
  samePattern :: V e -> V e -> Bool
  finalPatEnv :: PatVarEnv -> ModAlpha e -> PatVarEnv

instance Matchable Expr where
  equate      = equateExpr
  match       = matchExpr
  samePattern = samePatternExpr
  finalPatEnv = finalPatEnvExpr

equateExpr :: PatVar -> ModAlpha Expr -> MatchState Expr -> Maybe (MatchState Expr)
equateExpr pv (A benv e) ms = case hasMatch pv ms of
  Just sol
    | e == sol  -> Just ms
    | otherwise -> Nothing
  Nothing
    | noCaptured benv e          -> Just (addMatch pv e ms)
    | otherwise                  -> Nothing

traceWith f x = trace (f x) x

matchExpr :: ModAlpha Expr -> ModAlpha Expr -> (PatVarEnv, MatchState Expr) -> Maybe (PatVarEnv, MatchState Expr)
matchExpr pat@(A benv_pat e_pat) tar@(A benv_tar e_tar) (penv, ms) =
  -- traceWith (\res -> show ms ++ "  ->  matchExpr " ++ show pat ++ "   " ++ show tar ++ "  -> " ++ show (snd <$> res)) $
  case (e_pat, e_tar) of
  (Var v, _) | (penv', occ) <- canonOcc penv benv_pat v -> case occ of
    Pat pv -> (penv',) <$> equate pv tar ms
    Bound bv | Var v2 <- e_tar
             , Just bv2 <- lookupDBE v2 benv_tar
             , bv == bv2
             -> Just (penv, ms)
    Free v | Var v2 <- e_tar
           , Nothing <- lookupDBE v2 benv_tar
           , v == v2
           -> Just (penv, ms)
    _ -> Nothing
  (App f1 a1, App f2 a2) -> match (f1 <$ pat) (f2 <$ tar) (penv, ms) >>= match (a1 <$ pat) (a2 <$ tar)
  (Lam b1 e1, Lam b2 e2) -> match (A (extendDBE b1 benv_pat) e1) (A (extendDBE b2 benv_tar) e2) (penv, ms)
  (_, _) -> Nothing

samePatternExpr :: V Expr -> V Expr -> Bool
samePatternExpr a b = isJust (same a b)
  where
    same (V penv1 benv1 e1) (V penv2 benv2 e2) = case (e1, e2) of
      (Var v1, Var v2)
        | (penv1', occ1) <- canonOcc penv1 benv1 v1
        , (penv2', occ2) <- canonOcc penv2 benv2 v2
        , occ1 == occ2
        -> Just (penv1', penv2')
      (Lam b1 e1, Lam b2 e2) -> same (V penv1 (extendDBE b1 benv1) e1)
                                     (V penv2 (extendDBE b2 benv2) e2)
      (App f1 a1, App f2 a2) -> do
        (penv1', penv2') <- same (V penv1 benv1 f1) (V penv2 benv2 f2)
        same (V penv1' benv1 a1) (V penv2' benv2 a2)
      _ -> Nothing

finalPatEnvExpr :: PatVarEnv -> ModAlpha Expr -> PatVarEnv
finalPatEnvExpr penv a@(A benv e) = case e of
  Var v   -> fst $ canonOcc penv benv v
  App f a -> finalPatEnv (finalPatEnv penv (A benv f)) (A benv a)
  Lam b e -> finalPatEnv penv (A (extendDBE b benv) e)

type MExprMap = SEMap (V Expr) MExprMap'
data MExprMap' a
  = MEM { mem_pvar   :: PatVarMap a      -- Occurrence of a pattern var
        , mem_bvar   :: BoundVarMap a    -- Occurrence of a lam-bound var
        , mem_fvar   :: FreeVarMap a     -- Occurrence of a completely free var
        , mem_app    :: MExprMap (MExprMap a)
        , mem_lam    :: MExprMap a
        }

deriving instance Show v => Show (MExprMap' v)

class Matchable (MExpr tm) => MTrieMap tm where
  type MExpr tm :: Type
  emptyMTM :: tm a
  lookupPatMTM :: ModAlpha (MExpr tm) -> (MatchState (MExpr tm), tm a) -> Bag (MatchState (MExpr tm), a)
  alterPatMTM  :: ModAlpha (MExpr tm) -> (PatVarEnv -> XT a) -> PatVarEnv -> tm a -> tm a

instance (MTrieMap tm, MExpr tm ~ e) => MTrieMap (SEMap (V e) tm) where
  type MExpr (SEMap (V e) tm) = e
  emptyMTM = EmptySEM
  lookupPatMTM = lookupPatSEM
  alterPatMTM = alterPatSEM

instance MTrieMap MExprMap' where
  type MExpr MExprMap' = Expr
  emptyMTM = mkEmptyMExprMap
  lookupPatMTM = lookupPatMM
  alterPatMTM = alterPatMM

mkEmptyMExprMap :: MExprMap' a
mkEmptyMExprMap
  = MEM { mem_pvar = emptyPVM
        , mem_fvar = emptyFVM
        , mem_bvar = emptyBVM
        , mem_app  = emptyMTM
        , mem_lam  = emptyMTM }

lookupPatSEM
  :: MTrieMap tm => ModAlpha (MExpr tm) -> (MatchState (MExpr tm), SEMap (V (MExpr tm)) tm a) -> Bag (MatchState (MExpr tm), a)
lookupPatSEM a1@(A benv e) (ms, m) = case m of
  EmptySEM -> Bag.empty
  SingleSEM (V penv benv_pat e_pat) v
    | Just (_penv', ms') <- match (A benv_pat e_pat) a1 (penv, ms)
    -> Bag.single (ms', v)
    | otherwise
    -> Bag.empty
  MultiSEM m -> lookupPatMTM a1 (ms, m)

lookupPatMM :: ModAlpha Expr -> (MatchState Expr, MExprMap' a) -> Bag (MatchState Expr, a)
lookupPatMM ae@(A benv e) (ms, MEM { .. })
  = match mem_pvar `Bag.union` decompose e
  where
     match = Bag.mapMaybe match_one . Bag.fromList . IntMap.toList
     match_one (pv, x) = (, x) <$> equate pv ae ms

     decompose (Var v) = case lookupDBE v benv of
       Just bv -> lkBoundVarOcc bv (ms, mem_bvar)
       Nothing -> lkFreeVarOcc  v  (ms, mem_fvar)
     decompose (App e1 e2) = Bag.concatMap (lookupPatMTM (A benv e2)) $
                             lookupPatMTM (A benv e1) (ms, mem_app)
     decompose (Lam v e) = lookupPatMTM (A (extendDBE v benv) e) (ms, mem_lam)

alterPatSEM
  :: (MTrieMap tm)
  => ModAlpha (MExpr tm) -> (PatVarEnv -> XT a) -> PatVarEnv -> SEMap (V (MExpr tm)) tm a -> SEMap (V (MExpr tm)) tm a
alterPatSEM a1@(A benv e) f penv = go
  where
    penv' = finalPatEnv penv a1
    k1 = V penv' benv e
    go EmptySEM | Just v1 <- f penv' Nothing = SingleSEM k1 v1
                | otherwise                  = EmptySEM
    go m@(SingleSEM k2@(V penv2 benv2 e2) v2)
      | samePattern k1 k2 = case f penv' (Just v2) of
          Nothing -> EmptySEM
          Just v1 -> SingleSEM k1 v1
      | otherwise = case f penv' Nothing of
          Nothing -> m
          Just v1 -> MultiSEM $ alterPatMTM a1           (\_ _ -> Just v1) penv'
                              $ alterPatMTM (A benv2 e2) (\_ _ -> Just v2) penv2
                              $ emptyMTM
    go (MultiSEM m) = MultiSEM (alterPatMTM a1 f penv m)

alterPatMM :: ModAlpha Expr -> (PatVarEnv -> XT a) -> PatVarEnv -> MExprMap' a -> MExprMap' a
alterPatMM ae@(A benv e) f penv m@(MEM {..})
  = go e
  where
    go (Var v) | (penv', occ) <- canonOcc penv benv v = case occ of
      Pat pv      -> m { mem_pvar = alterPatVarOcc   pv (f penv') mem_pvar }
      Bound bv    -> m { mem_bvar = alterBoundVarOcc bv (f penv)  mem_bvar }
      Free fv     -> m { mem_fvar = alterFreeVarOcc  v  (f penv)  mem_fvar }
    go (App e1 e2) = m { mem_app  = alterPatMTM (A benv e1) (liftXT (alterPatMTM (A benv e2) f)) penv mem_app }
    go (Lam b e')  = m { mem_lam  = alterPatMTM (A (extendDBE b benv) e') f penv mem_lam }

liftXT :: MTrieMap tm
       => (PatVarEnv -> tm a -> tm a)
       -> PatVarEnv -> Maybe (tm a) -> Maybe (tm a)
liftXT alter penv Nothing  = Just (alter penv emptyMTM)
liftXT alter penv (Just m) = Just (alter penv m)


-- An ad-hoc definition of foldMM, because I don't want to define another
-- (duplicate) type class method

foldMM :: (v -> a -> a) -> a -> MExprMap v -> a
foldMM f z m = sem f z m
  where
    sem :: (v -> a -> a) -> a -> SEMap k MExprMap' v -> a
    sem f z EmptySEM = z
    sem f z (SingleSEM _ v) = f v z
    sem f z (MultiSEM m) = expr f z m
    expr :: (v -> a -> a) -> a -> MExprMap' v -> a
    expr f z MEM{..} =
      let !z1 = sem f z mem_lam in
      let !z2 = sem (\mem z -> z `seq` sem f z mem) z1 mem_app in
      let !z3 = foldFVM f mem_fvar z2 in
      let !z4 = foldBVM f mem_bvar z3 in
      let !z5 = foldPVM f mem_pvar z4 in
      z5


{- *********************************************************************
*                                                                      *
                  PatMap
*                                                                      *
********************************************************************* -}

type Match a = ([(Var, PatVar)], a)
type PatMap a = MExprMap (Match a)

emptyPatMap :: PatMap a
emptyPatMap = emptyMTM

insertPM :: forall a. [Var]   -- Pattern variables
                      -> Expr -- Pattern
                      -> a -> PatMap a -> PatMap a
insertPM pvars e x pm
  = alterPatMTM (deBruijnize e) f (emptyPVE (Set.fromList pvars)) pm
  where
    f :: PatVarEnv -> XT (Match a)
    f penv _ = Just (map inst_key pvars, x)
     -- The "_" means just overwrite previous value
     where
        inst_key :: Var -> (Var, PatVar)
        inst_key v = case lookupDBE v (pve_env penv) of
                         Nothing  -> error ("Unbound PatVar " ++ v)
                         Just pv -> (v, pv)

matchPM :: Expr -> PatMap a -> [ ([(Var,Expr)], a) ]
matchPM e pm
  = [ (map (lookup (getMatchingSubst ms)) prs, x)
    | (ms, (prs, x)) <- Bag.toList $ lookupPatMTM (deBruijnize e) (emptyMS, pm) ]
  where
    lookup :: PatSubst -> (Var, PatVar) -> (Var, Expr)
    lookup subst (v, pv) = (v, lookupPatSubst pv subst)

deletePM :: forall a. [Var]   -- Pattern variables
                      -> Expr -- Pattern
                      -> PatMap a -> PatMap a
deletePM pvars e pm
  = alterPatMTM (deBruijnize e) f (emptyPVE (Set.fromList pvars)) pm
  where
    f :: PatVarEnv -> XT (Match a)
    f _ _ = Nothing

mkPatMap :: [([Var], Expr, a)] -> PatMap a
mkPatMap = foldr (\(tmpl_vs, e, a) -> insertPM tmpl_vs e a) emptyPatMap

mkPatSet :: [([Var], Expr)] -> PatMap Expr
mkPatSet = mkPatMap . map (\(tmpl_vs, e) -> (tmpl_vs, e, e))

elemsPatSet :: PatMap Expr -> [([Var], Expr)]
elemsPatSet pm = foldMM (\(pks, e) pats -> (map fst pks, e):pats) [] pm

-- | See also 'exprMapToTree'
patMapToTree :: Show v => PatMap v -> Tree String
patMapToTree = Node "." . go_sem (\v -> [ Node (show v) [] ])
  where
    mk_node_nonempty lbl []       = []
    mk_node_nonempty lbl children = [ Node lbl children ]

    go_sem go_val EmptySEM = []
    go_sem go_val (SingleSEM k v) = mk_node_nonempty ("sing(" ++ show k ++ ")") (go_val v)
    go_sem go_val (MultiSEM mem) = go_mem go_val mem

    go_mem go_val MEM{..} = concat
      [ go_fvar go_val mem_fvar
      , go_bvar go_val mem_bvar
      , go_lam  go_val mem_lam
      , go_app  go_val mem_app
      ]

    go_bvar go_val bvm = [ Node ("bvar(" ++ show k ++ ")") (go_val v) | (k,v) <- IntMap.toList bvm ]
    go_fvar go_val m   = [ Node ("fvar(" ++ k      ++ ")") (go_val v) | (k,v) <- Map.toList m ]
    go_lit  go_val m   = [ Node ("lit("  ++ k      ++ ")") (go_val v) | (k,v) <- Map.toList m ]
    go_lam  go_val em  = mk_node_nonempty "λ" (go_sem go_val em)
    go_app :: (v -> Forest String) -> MExprMap (MExprMap v) -> Forest String
    go_app  go_val mem  = mk_node_nonempty "@" (go_sem (go_sem go_val) mem)

{- Doesn't work
{- *********************************************************************
*                                                                      *
                  Triangular substitutions
*                                                                      *
********************************************************************* -}

data TriSubst a = TS
  { ts_reprs :: PatVarMap PatVar
  , ts_rigid :: PatVarMap a }

emptyTS :: TriSubst a
emptyTS = TS { ts_reprs = emptyPVM, ts_rigid = emptyPVM }

-- | Returns the representative of the equality class of the given 'PatVar' and
-- a new 'TriSubst' with compressed paths.
findTS :: PatVar -> TriSubst a -> (PatVar, TriSubst a)
findTS v ts@TS{ts_reprs=reprs} = case IntMap.lookup v reprs of
  Nothing -> (v, ts)
  Just v'
    | let !(v'', ts') = findTS v' ts
    -> (v'', ts'{ts_reprs=IntMap.insert v v'' reprs})
{-# INLINE findTS #-}

-- | Merges the two equivalence classes represented by the given 'PatVar's.
unionTS :: PatVar -> PatVar -> TriSubst a -> TriSubst a
unionTS a b ts = ts{ts_reprs=IntMap.insert a b (ts_reprs ts)}
{-# INLINE unionTS #-}

lookupTS :: PatVar -> TriSubst a -> (Maybe a, TriSubst a)
lookupTS v ts = (IntMap.lookup v' (ts_rigid ts'), ts')
  where
    (v', ts') = findTS v ts
{-# INLINE lookupTS #-}

insertTS :: PatVar -> a -> TriSubst a -> TriSubst a
insertTS v x ts = ts'{ts_rigid=IntMap.insert v' x (ts_rigid ts')}
  where
    (v', ts') = findTS v ts

{- *********************************************************************
*                                                                      *
                  Unifying ExprMap
*                                                                      *
********************************************************************* -}

newtype UnifierState = US { us_subst :: TriSubst (V Expr) }

emptyUS :: UnifierState
emptyUS = US { us_subst = emptyTS }

--hasSolution :: PatVar -> UnifierState -> (Maybe (V Expr), UnifierState)
--hasSolution = coerce lookupTS

solve :: PatVar -> CExprDeep -> UnifierState -> Maybe UnifierState
solve = _

unifyEC :: V Expr -> CExprDeep -> UnifierState -> Maybe UnifierState
unifyEC = _

unifyCC :: CExprDeep -> CExprDeep -> UnifierState -> Maybe UnifierState
unifyCC = _
--unify ve1@(V benv1 penv1 e1) ve2@(V benv2 penv2 e2) us = case (e1, e2) of
--  (Var v1, e2') -> case  benv1 penv1 v1 of
--    Pat pv mb_penv
--      | Just _ <- mb_penv
--      -> error ("can't unify un-canonicalised PatVar " ++ show pv)
--      | (Just sol, us') <- hasSolution pv (us_left us)
--      -> unify sol ve2 us'
--      | pv `doesNotOccurIn` ve2
--      , noCaptured benv2 e2
--      -> Just us{us_subst=insertTS pv ve2 (us_left us)}
--      | otherwise
--      -> Nothing
--    occ1
--      | Var v2 <- e2' -> case viewOcc benv2 penv2 v2 of
--          occ2 | sameOcc occ1 occ2 -> Just us
--               | otherwise         -> Nothing
--    _ -> Nothing
--  (_, Var v2) -> unify ve2 ve1 us
--  (App f1 a1, App f2 a2) -> unify (f1 <$ ve1) (f2 <$ ve2) us >>= unify (a1 <$ ve1) (a2 <$ ve2)
--  (Lam v1 e1', Lam v2 e2') -> unify (add_bndr v1 (e1' <$ ve1)) (add_bndr v2 (e2' <$ ve2)) us
--  _ -> Nothing
--  where
--    add_bndr v (V benv penv a) = V (extendDBE v benv) penv a

-- | The "occurs check"
doesNotOccurIn :: PatVar -> V Expr -> Bool
doesNotOccurIn _ _ = True -- Fix when we do unification instead of matching

type UExprMap = SEMap UExprMap'
data UExprMap' a
  = UEM { uem_pvar :: PatVarMap a      -- Occurrence of a pattern var
        , uem_bvar :: BoundVarMap a    -- Occurrence of a lam-bound var
        , uem_fvar :: FreeVarMap a     -- Occurrence of a completely free var
        , uem_app  :: UExprMap (UExprMap a)
        , uem_lam  :: UExprMap a
        }

class MTrieMap tm => UTrieMap tm where
  lookupUnisTM :: ModAlpha Expr
               -> (PatVarEnv -> (UnifierState, a) -> Bag (UnifierState, b))
               ->  PatVarEnv -> (UnifierState, tm a) -> Bag (UnifierState, b)
  alterPatMTM :: ModAlpha Expr -> (PatVarEnv -> XT a) -> PatVarEnv -> tm a -> tm a

instance (UTrieMap tm, TrieKey tm ~ CExprDeep) => UTrieMap (SEMap tm) where
  lookupUnisTM = lookupUnisSEM
  alterPatMTM = alterPatSEM

instance TrieMap UExprMap' where
  type TrieKey UExprMap' = CExprDeep
  emptyTM = mkEmptyUExprMap

instance UTrieMap UExprMap' where
  lookupUnisTM = lookupUnisUM
  alterPatMTM = alterPatUM

deriving instance (Show v)
               => Show (UExprMap' v)

emptyUExprMap :: UExprMap a
emptyUExprMap = EmptySEM

mkEmptyUExprMap :: UExprMap' a
mkEmptyUExprMap
  = UEM { uem_pvar = emptyPVM
        , uem_fvar = emptyFVM
        , uem_bvar = emptyBVM
        , uem_app  = emptyUExprMap
        , uem_lam  = emptyUExprMap }

assocsSEM :: (tm a -> [(TrieKey tm, a)]) -> SEMap tm a -> [(TrieKey tm, a)]
assocsSEM _ EmptySEM        = []
assocsSEM _ (SingleSEM k x) = [(k, x)]
assocsSEM f (MultiSEM m)    = f m

assocsUM :: UExprMap' a -> [(CExprDeep, a)]
assocsUM UEM{..} = concat
  [ [ (Var (Pat pv), x)   | (pv, x) <- IntMap.toList uem_pvar ]
  , [ (Var (Bound bv), x) | (bv, x) <- IntMap.toList uem_bvar ]
  , [ (Var (Free v), x)   | (v, x)  <- Map.toList uem_fvar ]
  , [ (Lam () e, x)       | (e, x)  <- assocs_tm uem_lam ]
  , [ (App f a, x)        | (f, ma) <- assocs_tm uem_app, (a, x) <- assocs_tm ma ]
  ]
  where assocs_tm = assocsSEM assocsUM

lookupUnisSEM
  :: (UTrieMap tm, TrieKey tm ~ CExprDeep)
  => ModAlpha Expr
  -> (PatVarEnv -> (UnifierState, a) -> Bag (UnifierState, b))
  ->  PatVarEnv -> (UnifierState, SEMap tm a) -> Bag (UnifierState, b)
lookupUnisSEM a1@(A benv e) cont penv (us, m) = lk m
  where
    lk EmptySEM = Bag.empty
    lk (SingleSEM k2 v)
      | let k1 = V benv penv e
      , Just us' <- unifyEC k1 k2 us = cont (getFinalPatEnv k1) (us', v)
      | otherwise                    = Bag.empty
    lk (MultiSEM m) = lookupUnisTM a1 cont penv (us, m)

lookupUnisUM :: ModAlpha Expr
             -> (PatVarEnv -> (UnifierState, a) -> Bag (UnifierState, b))
             -> PatVarEnv -> (UnifierState, UExprMap' a) -> Bag (UnifierState, b)
lookupUnisUM k@(A benv e) cont penv (us, um@UEM{..})
  = match uem_pvar `Bag.union` decompose e
  where
     match = Bag.concatMap match_one . Bag.fromList . IntMap.toList
     match_one (pv, x) = case solve pv k us of
       Nothing -> Bag.empty
       Just us' -> cont penv' (us', x)
       where
         (penv', k) = canonDeep penv benv e

     decompose (Var v) | (penv', occ) <- canonOcc penv benv v = case occ of
       Pat pv   -> Bag.concatMap (cont penv) $ Bag.mapMaybe (\(e,x) -> (,x) <$> solve pv e us) $ Bag.fromList $ assocsUM um
       Bound bv -> Bag.concatMap (cont penv) (lkBoundVarOcc bv (us, uem_bvar))
       Free v   -> Bag.concatMap (cont penv) (lkFreeVarOcc  v  (us, uem_fvar))
     decompose (App e1 e2) = lookupUnisTM (A benv e1)
                                          (lookupUnisTM (A benv e2) cont)
                                          penv
                                          (us, uem_app)
     decompose (Lam v e) = lookupUnisTM (A (extendDBE v benv) e) cont penv (us, uem_lam)
     lookup_um a = lookupUnisSEM a

alterPatSEM
  :: (UTrieMap tm, TrieKey tm ~ V Expr)
  => ModAlpha Expr -> (PatVarEnv -> XT a) -> PatVarEnv -> SEMap tm a -> SEMap tm a
alterPatSEM a1@(A benv e) f penv = go
  where
    (penv', k1) = canonDeep penv benv e
    go EmptySEM | Just v1 <- f penv' Nothing = SingleSEM k1 v1
                | otherwise                  = EmptySEM
    go m@(SingleSEM k2 (penv2, v2))
      | k1 == k2 = case f penv' (Just v2) of
          Nothing -> EmptySEM
          Just v1 -> SingleSEM k1 v1
      | otherwise = case f penv' Nothing of
          Nothing -> m
          Just v1 -> MultiSEM $ alterPatMTM a1           (\_ _ -> Just v1) penv
                              $ alterPatMTM (A benv2 e2) (\_ _ -> Just v2) penv
                              $ emptyTM
    go (MultiSEM m) = MultiSEM (alterPatMTM a1 f penv m)

alterPatUM :: ModAlpha Expr -> (PatVarEnv -> XT a) -> PatVarEnv -> UExprMap' a -> UExprMap' a
alterPatUM ae@(A benv e) f penv m@(UEM {..})
  = go e
  where
    go (Var v) | (penv', occ) <- canonOcc penv benv v = case occ of
      Pat pv      -> m { uem_pvar = IntMap.alter (f penv') pv uem_pvar }
      Bound bv    -> m { uem_bvar = alterBoundVarOcc bv (f penv) uem_bvar }
      Free fv     -> m { uem_fvar = alterFreeVarOcc v (f penv) uem_fvar }
    go (App e1 e2) = m { uem_app  = alterPatMTM (A benv e1) (liftXTU (alterPatMTM (A benv e2) f)) penv uem_app }
    go (Lam v e')  = m { uem_lam  = alterPatMTM (A (extendDBE v benv) e') f penv uem_lam }


liftXTU :: (PatVarEnv -> UExprMap a -> UExprMap a)
        -> PatVarEnv -> Maybe (UExprMap a) -> Maybe (UExprMap a)
liftXTU alter penv Nothing  = Just (alter penv emptyUExprMap)
liftXTU alter penv (Just m) = Just (alter penv m)


{- *********************************************************************
*                                                                      *
                  UExprMap
*                                                                      *
********************************************************************* -}

type UMatch a = ([(Var, PatVar)], a)
type UniMap a = UExprMap (UMatch a)

emptyUniMap :: UniMap a
emptyUniMap = emptyUExprMap

--insertUM :: forall a. [Var]   -- Pattern variables
--                      -> Expr -- Pattern
--                      -> a -> UniMap a -> UniMap a
--insertUM pvars e x tm
--  = alterUni (deBruijnize e) f (emptyPVE (Set.fromList pvars)) tm
--  where
--    f :: PatVarEnv -> XT (UMatch a)
--    f penv _ = Just (map inst_key pvars, x)
--     -- The "_" means just overwrite previous value
--     where
--        inst_key :: Var -> (Var, UniVar)
--        inst_key v = case lookupDBE v (pve_env penv) of
--                         Nothing  -> error ("Unbound UniVar " ++ v)
--                         Just pv -> (v, pv)
--
--matchUM :: Expr -> UniMap a -> [ ([(Var,Expr)], a) ]
--matchUM e tm
--  = [ (map (lookup (getUnifyingSubst ms)) prs, x)
--    | (ms, (prs, x)) <- Bag.toList $ lookupInsts (deBruijnize e) (emptyUS, tm) ]
--  where
--    lookup :: PatSubst -> (Var, PatVar) -> (Var, Expr)
--    lookup subst (v, pv) = (v, lookupUniSubst pv subst)
--
--mkUniMap :: [([Var], Expr, a)] -> UniMap a
--mkUniMap = foldr (\(tmpl_vs, e, a) -> insertUM tmpl_vs e a) emptyUExprMap
--
--mkUniSet :: [([Var], Expr)] -> UniMap Expr
--mkUniSet = mkUniMap . map (\(tmpl_vs, e) -> (tmpl_vs, e, e))
-}

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

instance (TrieMapLens tm, TrieKey tm ~ k) => TrieMapLens (SEMap k tm) where
  atTM = atSEM
  nullTM = nullSEM

nullSEM :: TrieMapLens tm => SEMap k tm v -> Bool
nullSEM EmptySEM      = True
-- nullSEM (MultiSEM tm) = nullTM tm -- Invariant: MultiSEM is never empty
nullSEM _             = False

atSEM :: TrieMapLens tm => TrieKey tm -> Lens' (SEMap (TrieKey tm) tm v) (Maybe v)
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

atExpr :: ModAlpha Expr -> Lens' (ExprMap' v) (Maybe v)
atExpr (A dbe e) = case e of
  Var x     -> case lookupDBE x dbe of
                  Just bv -> lens_em_bvar . at bv -- NB: at from microlens's `At (IntMap v)` instance
                  Nothing -> lens_em_fvar . at x
  App e1 e2 -> lens_em_app . atTM (A dbe e1) . nonEmpty . atTM (A dbe e2)
  Lam x e   -> lens_em_lam . atTM (A (extendDBE x dbe) e)

-- We would have to define these microlens instances for every TrieMapLens. It's
-- weirdly redundant, I'm only doing so for ExprMap' here:
type instance Index (ExprMap' v) = ModAlpha Expr
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
  ppr (A bv a) = text "D" PP.<> braces (sep [ppr bv, ppr a])

instance (Pretty v, Pretty (tm v), Pretty (TrieKey tm), TrieKey tm ~ k)
      => Pretty (SEMap k tm v) where
  ppr EmptySEM        = text "EmptySEM"
  ppr (SingleSEM k v) = text "SingleSEM" <+> ppr k <+> ppr v
  ppr (MultiSEM tm)   = ppr tm

instance Pretty a => Pretty (ExprMap' a) where
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "em_bvar =" <+> ppr em_bvar
                    , text "em_fvar =" <+> ppr em_fvar
                    , text "em_app ="  <+> ppr em_app
                    , text "em_lam ="  <+> ppr em_lam ])

{-
instance Pretty a => Pretty (PatMap a) where
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "mem_pvar =" <+> ppr mem_pvar
                    , text "mem_bvar =" <+> ppr mem_bvar
                    , text "mem_fvar =" <+> ppr mem_fvar
                    , text "mem_app =" <+> ppr mem_app
                    , text "mem_lam =" <+> ppr mem_lam ])
-}
