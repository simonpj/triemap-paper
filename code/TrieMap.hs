{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables,
             StandaloneDeriving, UndecidableInstances,
             LambdaCase #-}

{-# LANGUAGE BangPatterns, StrictData #-} -- for benchmarks

{-# OPTIONS_GHC -Wincomplete-patterns #-}

-- | This module presents
--
--   * The generic triemap transformers 'SEMap' and 'ListMap' from the paper
--   * A trie map 'ExprMap' that is modelled after @ExprLMap@ from the paper,
--     with an additional 'Lit' constructor in the 'Expr' type.
--   * The matching trie map type 'MExprMap' from the paper.
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
                   Tests
*                                                                      *
********************************************************************* -}

-- Here are a few definitions intended for playing around with 'MExprMap'.
-- It makes use of the 'Read' instance of 'Expr' to parse expressions.
-- See the haddock on that 'Read' instance for more details on what
-- assumptions the parser makes.

item1, item2, item3, item4, item5 :: (String, [Var], Expr)
item1 = ("item1", ["a"], read "a I")
item2 = ("item2", ["a"], read "a a")
item3 = ("item3", [],    read "I I")
item4 = ("item4", ["a", "b"], read "b a")
item5 = ("item5", ["d"], read "I d")

ty1, ty2, ty3 :: Expr
ty1 = read "I I"
ty2 = read "C C"
ty3 = read "C I"

ins :: MExprMap String -> (String, [Var], Expr) -> MExprMap String
ins m (s,tvs,key) = insertMExprMap tvs key s m

initM :: [(String,[Var],Expr)] -> MExprMap String
initM items = foldl ins emptyMExprMap items

{- *********************************************************************
*                                                                      *
                   Expr
*                                                                      *
********************************************************************* -}

type Var    = String
type Lit    = String

data Expr = Var Var
          | App Expr Expr
          | Lam Var Expr
          | Lit Lit

anyFreeVarsOfExpr :: (Var -> Bool) -> Expr -> Bool
-- True if 'p' returns True of any free variable
-- of the type; False otherwise
anyFreeVarsOfExpr p ty
  = go Set.empty ty
  where
    go bvs (Var tv) | tv `Set.member` bvs = False
                        | otherwise           = p tv
    go bvs (App t1 t2)    = go bvs t1 || go bvs t2
    go bvs (Lam tv ty) = go (Set.insert tv bvs) ty
    go _   (Lit _)      = False

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
type BoundVarMap = IntMap

emptyBVM :: BoundVarMap a
emptyBVM = IntMap.empty

lookupBVM :: BoundVar -> BoundVarMap a -> Maybe a
lookupBVM = IntMap.lookup

extendBVM :: BoundVar -> a -> BoundVarMap a -> BoundVarMap a
extendBVM = IntMap.insert

foldBVM :: (v -> a -> a) -> BoundVarMap v -> a -> a
foldBVM k m z = foldr k z m

lkBoundVarOcc :: BoundVar -> (PatSubst, BoundVarMap a) -> Bag (PatSubst, a)
lkBoundVarOcc var (tsubst, env) = case lookupBVM var env of
                                     Just x  -> Bag.single (tsubst,x)
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

eqDBExpr (A env1 (Var tv1)) (A env2 (Var tv2))
  = case (lookupDBE tv1 env1, lookupDBE tv2 env2) of
      (Just bvi1, Just bvi2) -> bvi1 == bvi2
      (Nothing,   Nothing)   -> tv1 == tv2
      _                      -> False

eqDBExpr (A _ (Lit tc1)) (A _ (Lit tc2))
  = tc1 == tc2

eqDBExpr (A env1 (Lam tv1 t1)) (A env2 (Lam tv2 t2))
  = eqDBExpr (A (extendDBE tv1 env1) t1)
           (A (extendDBE tv2 env2) t2)

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
exprTag Lit{} = 3
{-# INLINE exprTag #-}

cmpDBExpr :: ModAlpha Expr -> ModAlpha Expr -> Ordering
cmpDBExpr (A _ (Lit l1))       (A _ (Lit l2))
  = compare l1 l2

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
lookupFVM env tv = Map.lookup tv env

extendFVM :: FreeVarMap a -> Var -> a -> FreeVarMap a
extendFVM env tv val = Map.insert tv val env

foldFVM :: (v -> a -> a) -> FreeVarMap v -> a -> a
foldFVM k m z = foldr k z m

alterFreeVarOcc :: Var -> XT a -> FreeVarMap a -> FreeVarMap a
alterFreeVarOcc tv xt tm = Map.alter xt tv tm

lkFreeVarOcc :: Var -> (PatSubst, FreeVarMap a) -> Bag (PatSubst, a)
lkFreeVarOcc var (tsubst, env) = case Map.lookup var env of
                                    Just x  -> Bag.single (tsubst,x)
                                    Nothing -> Bag.empty

{- *********************************************************************
*                                                                      *
                   Pattern variables
*                                                                      *
********************************************************************* -}

type PatVar    = Int
type PatVarMap = IntMap

data PatSubst = PatVarMap Expr -- Maps PatVar :-> Expr

emptyPatSubst :: PatSubst
emptyPatSubst = IntMap.empty

lookupPatSubst :: PatVar -> PatSubst -> Expr
lookupPatSubst key subst
  = case IntMap.lookup key subst of
      Just ty -> ty
      Nothing -> error ("lookupPatSubst " ++ show key)

data ModPatKeys a = P !(Set Var) !DeBruijnEnv !a

canonPV :: ModPatKeys Var -> ModPatKeys (Maybe PatVar)
canonPV (P pvs dbe v)
  | not (v `Set.member` pvs)   = P pvs dbe               Nothing
  | Just pv <- lookupDBE v dbe = P pvs dbe               (Just pv)
  | otherwise                  = P pvs (extendDBE v dbe) (Just (dbe_next dbe))


newtype Ctx a = C (ModAlpha (ModPatKeys a))

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

data SEMap tm v
  = EmptySEM
  | SingleSEM (TrieKey tm) v
  | MultiSEM  (tm v)

deriving instance (Show v, Show (TrieKey tm), Show (tm v))
               => Show (SEMap tm v)


instance TrieMap tm => TrieMap (SEMap tm) where
  type TrieKey (SEMap tm) = TrieKey tm
  emptyTM     = EmptySEM
  lookupTM    = lookupSEM
  alterTM     = alterSEM
  unionWithTM = unionWithSEM
  foldTM      = foldSEM
  -- fromListWithTM = fromListWithSEM

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
unionWithSEM f (MultiSEM tm)   (SingleSEM k v)
  = MultiSEM $ alterTM k (\_ -> Just v) tm
unionWithSEM f (SingleSEM k v) (MultiSEM tm)
  = MultiSEM $ alterTM k xt tm
  where
    xt Nothing = Just v
    xt old     = old
unionWithSEM f (MultiSEM tm1)  (MultiSEM tm2)
  = MultiSEM $ unionWithTM f tm1 tm2

foldSEM :: TrieMap tm => (v -> a -> a) -> SEMap tm v -> a -> a
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

type ListMap tm = SEMap (ListMap' tm)

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


type ExprMap = SEMap ExprMap'

data ExprMap' a
  = EM { em_bvar :: BoundVarMap a    -- Occurrence of a forall-bound tyvar
       , em_fvar :: FreeVarMap a     -- Occurrence of a completely free tyvar

       , em_app  :: ExprMap (ExprMap a)
       , em_lit  :: Map Lit a
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
       , em_lit  = Map.empty
       , em_lam  = emptyExprMap }

lookupExpr :: ModAlpha Expr -> ExprMap' v -> Maybe v
lookupExpr (A dbe e) (EM { .. })
  = case e of
      Var x     -> case lookupDBE x dbe of
                     Just bv -> em_bvar |> lookupBVM bv
                     Nothing -> em_fvar |> Map.lookup x
      App e1 e2 -> em_app |>  lookupTM (A dbe e1)
                          >=> lookupTM (A dbe e2)
      Lit lit   -> em_lit |> Map.lookup lit
      Lam x e   -> em_lam |> lookupTM (A (extendDBE x dbe) e)

alterExpr :: ModAlpha Expr -> XT v -> ExprMap' v -> ExprMap' v
alterExpr (A dbe e) xt m@(EM {..})
  = case e of
      Var x -> case lookupDBE x dbe of
                  Just bv -> m { em_bvar = alterBoundVarOcc bv xt em_bvar }
                  Nothing -> m { em_fvar = alterFreeVarOcc  x  xt em_fvar }

      Lit lit   -> m { em_lit = em_lit |> Map.alter xt lit }
      App e1 e2 -> m { em_app = em_app |> alterTM (A dbe e1) |>> alterTM (A dbe e2) xt }
      Lam x e   -> m { em_lam = em_lam |> alterTM (A (extendDBE x dbe) e) xt }

unionWithExpr :: (v -> v -> v) -> ExprMap' v -> ExprMap' v -> ExprMap' v
unionWithExpr f m1 m2
  = EM { em_bvar = IntMap.unionWith f (em_bvar m1) (em_bvar m2)
       , em_fvar = Map.unionWith f (em_fvar m1) (em_fvar m2)
       , em_lit  = Map.unionWith f (em_lit m1) (em_lit m2)
       , em_app  = unionWithTM (unionWithTM f) (em_app m1) (em_app m2)
       , em_lam  = unionWithTM f (em_lam m1) (em_lam m2) }

foldExpr :: forall a v. (v -> a -> a) -> ExprMap' v -> a -> a
foldExpr f (EM {..}) z
  = let !z1 = foldTM f em_lam z in
    let !z2 = foldr f z1 em_lit in
    let !z3 = foldTM (\em z -> z `seq` foldTM f em z) em_app z2 in
    let !z4 = foldFVM f em_fvar z3 in
    foldBVM f em_bvar z4


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
      [ go_lit  go_val em_lit
      , go_fvar go_val em_fvar
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
                  Matching ExprMap
*                                                                      *
********************************************************************* -}

data MExprMapX a
  = EmptyMEM
  | MEM { mem_tvar   :: Maybe a          -- First occurrence of a template tyvar
        , mem_xvar   :: PatVarMap a       -- Subsequent occurrence of a template tyvar

        , mem_bvar   :: BoundVarMap a    -- Occurrence of a lam-bound tyvar
        , mem_fvar   :: FreeVarMap a     -- Occurrence of a completely free tyvar

        , mem_fun    :: MExprMapX (MExprMapX a)
        , mem_tycon  :: Map Lit a
        , mem_lam    :: MExprMapX a
        }

deriving instance (Show v)
               => Show (MExprMapX v)

emptyMExprMapX :: MExprMapX a
emptyMExprMapX = EmptyMEM

mkEmptyMExprMapX :: MExprMapX a
mkEmptyMExprMapX
  = MEM { mem_tvar   = Nothing
        , mem_fvar   = emptyFVM
        , mem_xvar   = IntMap.empty
        , mem_bvar   = emptyBVM
        , mem_fun    = emptyMExprMapX
        , mem_tycon  = Map.empty
        , mem_lam    = emptyMExprMapX }

lkT :: ModAlpha Expr -> (PatSubst, MExprMapX a) -> Bag (PatSubst, a)
lkT (A dbe ty) (psubst, EmptyMEM)
  = Bag.empty
lkT (A dbe ty) (psubst, MEM { .. })
  = tmpl_var_bndr `Bag.union` rest
  where
     rest = tmpl_var_occs `Bag.union` go ty
     no_more_specific_matches = not (Bag.any is_more_specific rest)
     is_more_specific (psubst', _) = ts_next psubst' > ts_next psubst

     go (Var tv)
       | Just bv <- lookupDBE tv dbe = lkBoundVarOcc bv (psubst, mem_bvar)
       | otherwise                   = lkFreeVarOcc  tv (psubst, mem_fvar)
     go (App t1 t2) = Bag.concatMap (lkT (A dbe t2)) $
                      lkT (A dbe t1) (psubst, mem_fun)
     go (Lit tc)    = lkTC tc psubst mem_tycon

     go (Lam tv ty) = lkT (A (extendDBE tv dbe) ty) (psubst, mem_lam)

     tmpl_var_bndr | Just x <- mem_tvar
                   , no_more_specific_matches    -- This one line does overlap!
                   , noCaptured dbe ty
                   = Bag.single (extendPatSubst ty psubst, x)
                   | otherwise
                   = Bag.empty

     tmpl_var_occs = Bag.fromList [ (psubst, x)
                                  | (tmpl_var, x) <- IntMap.toList mem_xvar
                                  , deBruijnize (lookupPatSubst tmpl_var psubst)
                                    `eqDBExpr` (A dbe ty)
                                  ]

lkTC :: Lit -> PatSubst -> Map Lit a -> Bag (PatSubst, a)
lkTC tc psubst tc_map = case Map.lookup tc tc_map of
                           Nothing -> Bag.empty
                           Just x  -> Bag.single (psubst,x)

xtT :: Set Var -> ModAlpha Expr
    -> (PatVar -> XT a)
    -> PatVar -> MExprMapX a -> MExprMapX a
xtT pvs (A dbe ty) f tkeys EmptyMEM
  = xtT pvs (A dbe ty) f tkeys mkEmptyMExprMapX

xtT pvs (A dbe ty) f tkeys m@(MEM {..})
  = go ty
  where
   go (Var tv)
      -- Second or subsequent occurrence of a template tyvar
      | Just xv <- lookupDBE tv tkeys  = m { mem_xvar = IntMap.alter (f tkeys) xv mem_xvar }

      -- First occurrence of a template tyvar
      | tv `Set.member` pvs = m { mem_tvar = f (extendDBE tv tkeys) mem_tvar  }

      -- Occurrence of a lam-bound var
      | Just bv <- lookupDBE tv dbe = m { mem_bvar = alterBoundVarOcc bv (f tkeys) mem_bvar }

      -- A completely free variable
      | otherwise = m { mem_fvar = alterFreeVarOcc  tv (f tkeys) mem_fvar }

   go (Lit tc)  = m { mem_tycon = xtTC tc (f tkeys) mem_tycon }
   go (App t1 t2) = m { mem_fun   = xtT pvs (A dbe t1)
                                         (liftXT (xtT pvs (A dbe t2) f))
                                         tkeys mem_fun }
   go (Lam tv ty) = m { mem_lam = xtT pvs (A (extendDBE tv dbe) ty)
                                             f tkeys mem_lam }


xtTC :: Lit -> XT a -> Map Lit a ->  Map Lit a
xtTC tc f m = Map.alter f tc m

liftXT :: (PatVar -> MExprMapX a -> MExprMapX a)
        -> PatVar -> Maybe (MExprMapX a) -> Maybe (MExprMapX a)
liftXT insert tkeys Nothing  = Just (insert tkeys emptyMExprMapX)
liftXT insert tkeys (Just m) = Just (insert tkeys m)


{- *********************************************************************
*                                                                      *
                  MExprMap
*                                                                      *
********************************************************************* -}

type Match a = ([(Var, PatVar)], a)
type MExprMap a = MExprMapX (Match a)

emptyMExprMap :: MExprMap a
emptyMExprMap = emptyMExprMapX

insertMExprMap :: forall a. [Var]   -- Pattern variables
                         -> Expr    -- Patern
                         -> a -> MExprMap a -> MExprMap a
insertMExprMap tmpl_tvs ty x tm
  = xtT tmpl_set (deBruijnize ty) f emptyDBE tm
  where
    tmpl_set = Set.fromList tmpl_tvs

    f :: PatVar -> XT (Match a)
    f tkeys _ = Just (map inst_key tmpl_tvs, x)
     -- The "_" means just overwrite previous value
     where
        inst_key :: Var -> (Var, PatVar)
        inst_key tv = case lookupDBE tv tkeys of
                         Nothing  -> error ("Unbound tmpl var " ++ tv)
                         Just key -> (tv, key)

lookupMExprMap :: Expr -> MExprMap a -> [ ([(Var,Expr)], a) ]
lookupMExprMap ty tm
  = [ (map (lookup psubst) prs, x)
    | (psubst, (prs, x)) <- Bag.toList $ lkT (deBruijnize ty) (emptyPatSubst, tm) ]
  where
    lookup :: PatSubst -> (Var, PatVar) -> (Var, Expr)
    lookup psubst (tv, key) = (tv, lookupPatSubst key psubst)

mkMExprMap :: [([Var], Expr, a)] -> MExprMap a
mkMExprMap = foldr (\(tmpl_vs, e, a) -> insertMExprMap tmpl_vs e a) emptyMExprMap

mkMExprSet :: [([Var], Expr)] -> MExprMap Expr
mkMExprSet = mkMExprMap . map (\(tmpl_vs, e) -> (tmpl_vs, e, e))

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

instance TrieMapLens tm => TrieMapLens (SEMap tm) where
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

lens_em_lit :: Lens' (ExprMap' a) (Map Lit a)
lens_em_lit xf em@(EM { .. }) = xf em_lit <&> \lit' -> em { em_lit = lit' }

lens_em_lam :: Lens' (ExprMap' a) (ExprMap a)
lens_em_lam xf em@(EM { .. }) = xf em_lam <&> \lam' -> em { em_lam = lam' }

nullExpr :: ExprMap' v -> Bool
nullExpr (EM {..}) =  Prelude.null em_fvar && Prelude.null em_bvar
                   && Prelude.null em_lit && nullTM em_app && nullTM em_lam

atExpr :: ModAlpha Expr -> Lens' (ExprMap' v) (Maybe v)
atExpr (A dbe e) = case e of
  Var x     -> case lookupDBE x dbe of
                  Just bv -> lens_em_bvar . at bv -- NB: at from microlens's `At (IntMap v)` instance
                  Nothing -> lens_em_fvar . at x
  Lit lit   -> lens_em_lit . at lit
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
  showsPrec _ (Lit l)      = showString l
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
        pure $ if isLower (head v)
          then Var v
          else Lit v
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
  ppr ty = text (show ty)

instance Pretty DeBruijnEnv where
  ppr (DBE { .. }) = text "DBE" PP.<>
                     braces (sep [ text "dbe_next =" <+> ppr dbe_next
                                 , text "dbe_env =" <+> ppr dbe_env ])

instance (Pretty a) => Pretty (ModAlpha a) where
  ppr (A bv a) = text "D" PP.<> braces (sep [ppr bv, ppr a])

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
                    , text "em_lit ="  <+> ppr em_lit
                    , text "em_lam ="  <+> ppr em_lam ])

{-
instance Pretty a => Pretty (MExprMap a) where
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "mem_tvar =" <+> ppr mem_tvar
                    , text "mem_xvar =" <+> ppr mem_xvar
                    , text "mem_bvar =" <+> ppr mem_bvar
                    , text "mem_fvar =" <+> ppr mem_fvar
                    , text "mem_fun =" <+> ppr mem_fun
                    , text "mem_tycon =" <+> ppr mem_tycon
                    , text "mem_lam =" <+> ppr mem_lam ])
-}
