{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables, LambdaCase,
             StandaloneDeriving, UndecidableInstances #-}

{-# OPTIONS_GHC -Wincomplete-patterns #-}

module GenTrieMap where

import Data.List( foldl' )
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.IntMap as IntMap
import Data.Kind
import Control.Monad
import Data.Functor( (<$>), (<&>) )
import Data.Functor.Const( Const(Const, getConst) )
import Data.Functor.Identity( Identity(Identity, runIdentity) )
import Data.Maybe( isJust )

import Text.PrettyPrint as PP
import Debug.Trace
import Data.Char
import qualified Text.Read as Read
import qualified Text.ParserCombinators.ReadP as ReadP
import Data.Tree

{- *********************************************************************
*                                                                      *
                   Tests
*                                                                      *
********************************************************************* -}

item1, item2, item3, item4 :: (String, [Var], Expr)
item1 = ("item1", ["a"], read "a I")
item2 = ("item2", ["a"], read "a a")
item3 = ("item3", [],    read "I I")
item4 = ("item4", ["a", "b"], read "b a")

ty1, ty2, ty3 :: Expr
ty1 = read "I I"
ty2 = read "C C"
ty3 = read "C I"

-- ins :: MExprMap String -> (String, [Var], Expr) -> MExprMap String
--ins m (s,tvs,ty) = insertMExprMap tvs ty s m

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
               Bound variables
*                                                                      *
********************************************************************* -}

type BoundVarKey = Int

data DeBruijnEnv = DBE { dbe_next :: !BoundVarKey
                       , dbe_env  :: Map.Map Var BoundVarKey }

emptyDBE :: DeBruijnEnv
emptyDBE = DBE { dbe_next = 0, dbe_env = Map.empty }

extendDBE :: Var -> DeBruijnEnv -> DeBruijnEnv
extendDBE tv (DBE { dbe_next = bv, dbe_env = env })
  = DBE { dbe_next = bv+1, dbe_env = Map.insert tv bv env }

lookupDBE :: Var -> DeBruijnEnv -> Maybe BoundVar
lookupDBE v (DBE { dbe_env = env }) = Map.lookup v env

-- | @DeBruijn a@ represents @a@ modulo alpha-renaming.  This is achieved
-- by equipping the value with a 'DeBruijnEnv', which tracks an on-the-fly deBruijn
-- numbering.  This allows us to define an 'Eq' instance for @DeBruijn a@, even
-- if this was not (easily) possible for @a@.  Note: we purposely don't
-- export the constructor.  Make a helper function if you find yourself
-- needing it.
data DeBruijn a = D DeBruijnEnv a

-- | Synthesizes a @DeBruijn a@ from an @a@, by assuming that there are no
-- bound binders (an empty 'DeBruijnEnv').  This is usually what you want if there
-- isn't already a 'DeBruijnEnv' in scope.
deBruijnize :: a -> DeBruijn a
deBruijnize e = D emptyDBE e

instance Eq (DeBruijn a) => Eq (DeBruijn [a]) where
    D _   []     == D _    []       = True
    D env (x:xs) == D env' (x':xs') = D env x  == D env' x' &&
                                      D env xs == D env' xs'
    _            == _               = False

instance Eq (DeBruijn a) => Eq (DeBruijn (Maybe a)) where
    D _   Nothing  == D _    Nothing   = True
    D env (Just x) == D env' (Just x') = D env x  == D env' x'
    _              == _                = False

noCaptured :: DeBruijnEnv -> Expr -> Bool
-- True iff no free var of the type is bound by DeBruijnEnv
noCaptured dbe ty
  = not (anyFreeVarsOfExpr captured ty)
  where
    captured tv = isJust (lookupDBE tv dbe)

instance Eq (DeBruijn Expr) where
  (==) = eqDBExpr

eqDBExpr :: DeBruijn Expr -> DeBruijn Expr -> Bool
eqDBExpr (D env1 (App s1 t1)) (D env2 (App s2 t2))
  = eqDBExpr (D env1 s1) (D env2 s2) &&
    eqDBExpr (D env1 t1) (D env2 t2)

eqDBExpr (D env1 (Var tv1)) (D env2 (Var tv2))
  = case (lookupDBE tv1 env1, lookupDBE tv2 env2) of
      (Just bvi1, Just bvi2) -> bvi1 == bvi2
      (Nothing,   Nothing)   -> tv1 == tv2
      _                      -> False

eqDBExpr (D _ (Lit tc1)) (D _ (Lit tc2))
  = tc1 == tc2

eqDBExpr (D env1 (Lam tv1 t1)) (D env2 (Lam tv2 t2))
  = eqDBExpr (D (extendDBE tv1 env1) t1)
           (D (extendDBE tv2 env2) t2)

eqDBExpr _ _ = False

instance Show (DeBruijn Expr) where
  show (D _ e) = show e

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

cmpDBExpr :: DeBruijn Expr -> DeBruijn Expr -> Ordering
cmpDBExpr (D _ (Lit l1))       (D _ (Lit l2))
  = compare l1 l2

cmpDBExpr (D env1 (App f1 a1)) (D env2 (App f2 a2))
  = cmpDBExpr (D env1 f1) (D env2 f2) Prelude.<> cmpDBExpr (D env1 a1) (D env2 a2)

cmpDBExpr (D env1 (Lam v1 e1)) (D env2 (Lam v2 e2))
  = cmpDBExpr (D (extendDBE v1 env1) e1) (D (extendDBE v2 env2) e2)

cmpDBExpr (D env1 (Var v1))    (D env2 (Var v2))
  = case (lookupDBE v1 env1, lookupDBE v2 env2) of
      (Just bvi1, Just bvi2) -> compare bvi1 bvi2
      (Nothing,   Nothing)   -> compare v1 v2
      (Just _,    Nothing)   -> GT
      (Nothing,   Just _)    -> LT
cmpDBExpr (D _ e1) (D _ e2)
  = compare (exprTag e1) (exprTag e2)

instance Ord Expr where
  compare a b = cmpDBExpr (deBruijnize a) (deBruijnize b)

{- *********************************************************************
*                                                                      *
                   Template variables
*                                                                      *
********************************************************************* -}

{-
type PatVar    = Var
type PatVarSet = Set.Set Var
type PatKey    = BoundVarKey
type PatKeys   = DeBruijnEnv  -- Maps PatVar :-> PatKey

type PatOccs a = [(PatKey,a)]

xtPatVarOcc :: PatKey -> XT a -> PatOccs a -> PatOccs a
xtPatVarOcc key f []
  = xtCons key (f Nothing) []
xtPatVarOcc key f ((key1,x):prs)
  | key == key1 = xtCons key (f (Just x)) prs
  | otherwise   = (key1,x) : xtPatVarOcc key f prs

xtCons :: PatKey -> Maybe a -> PatOccs a -> PatOccs a
xtCons _   Nothing  tmpl_occs = tmpl_occs
xtCons key (Just x) tmpl_occs = (key,x) : tmpl_occs

---------------
data PatSubst = TS { ts_subst :: IntMap.IntMap Expr     -- Maps PatKey -> Expr
                    , ts_next  :: PatKey }

emptyPatSubst :: PatSubst
emptyPatSubst = TS { ts_subst = IntMap.empty
                    , ts_next  = 0 }

lookupPatSubst :: PatKey -> PatSubst -> Expr
lookupPatSubst key (TS { ts_subst = subst })
  = case IntMap.lookup key subst of
      Just ty -> ty
      Nothing -> error ("lookupPatSubst " ++ show key)

extendPatSubst :: Expr -> PatSubst -> PatSubst
extendPatSubst ty (TS { ts_subst = subst, ts_next = n })
  = TS { ts_subst = IntMap.insert n ty subst
       , ts_next  = n+1 }
-}

{- *********************************************************************
*                                                                      *
               Bound variables
*                                                                      *
********************************************************************* -}

type BoundVar = BoundVarKey  -- Bound variables are deBruijn numbered
type BoundVarMap a = IntMap.IntMap a

emptyBVM :: BoundVarMap a
emptyBVM = IntMap.empty

lookupBVM :: BoundVar -> BoundVarMap a -> Maybe a
lookupBVM = IntMap.lookup

extendBVM :: BoundVar -> a -> BoundVarMap a -> BoundVarMap a
extendBVM = IntMap.insert

foldBVM :: (v -> a -> a) -> BoundVarMap v -> a -> a
foldBVM k m z = foldr k z m

{-
lkBoundVarOcc :: BoundVar -> (PatSubst, BoundVarMap a) -> [(PatSubst, a)]
lkBoundVarOcc var (tsubst, env) = case lookupBVM var env of
                                     Just x  -> [(tsubst,x)]
                                     Nothing -> []
-}

alterBoundVarOcc :: Functor f => BoundVar -> XT f a -> BoundVarMap a -> f (BoundVarMap a)
alterBoundVarOcc tv xt tm = IntMap.alterF xt tv tm



{- *********************************************************************
*                                                                      *
                   Free variables
*                                                                      *
********************************************************************* -}

type FreeVarMap a = Map.Map Var a

emptyFVM :: FreeVarMap a
emptyFVM = Map.empty

lookupFVM :: FreeVarMap a -> Var -> Maybe a
lookupFVM env tv = Map.lookup tv env

extendFVM :: FreeVarMap a -> Var -> a -> FreeVarMap a
extendFVM env tv val = Map.insert tv val env

foldFVM :: (v -> a -> a) -> FreeVarMap v -> a -> a
foldFVM k m z = foldr k z m

alterFreeVarOcc :: Functor f => Var -> XT f a -> FreeVarMap a -> f (FreeVarMap a)
alterFreeVarOcc tv xt tm = Map.alterF xt tv tm

{-
lkFreeVarOcc :: Var -> (PatSubst, FreeVarMap a) -> [(PatSubst, a)]
lkFreeVarOcc var (tsubst, env) = case Map.lookup var env of
                                    Just x  -> [(tsubst,x)]
                                    Nothing -> []
-}

{- *********************************************************************
*                                                                      *
                  The TrieMap class
*                                                                      *
********************************************************************* -}

type XT f v = XF f (Maybe v)   -- How to alter a non-existent elt (Nothing)
                               --               or an existing elt (Just)
type XF f a = a -> f a

class Eq (TrieKey tm) => TrieMap tm where
   type TrieKey tm :: Type
   emptyTM  :: tm v
   alterTM  :: Functor f => TrieKey tm -> XT f v -> XF f (tm v)
   foldTM   :: (v -> a -> a) -> tm v -> a -> a

--   mapTM    :: (a->b) -> tm a -> tm b
--   filterTM :: (a -> Bool) -> tm a -> tm a
--   unionTM  ::  tm a -> tm a -> tm a

lookupTM :: forall tm v. TrieMap tm => TrieKey tm -> tm v -> Maybe v
lookupTM k tm = getConst (alterTM k xt tm)
  where
    xt :: Maybe v -> Const (Maybe v) (Maybe v)
    xt = Const

insertTM :: forall tm v. TrieMap tm => TrieKey tm -> v -> tm v -> tm v
insertTM k v tm = runIdentity (alterTM k xt tm)
   where
     xt :: Maybe v -> Identity (Maybe v)
     xt _ = Identity (Just v)

-- Recall that
--   Control.Monad.(>=>) :: (a -> Maybe b) -> (b -> Maybe c) -> a -> Maybe c

(>.>) :: (a -> b) -> (b -> c) -> a -> c
-- Reverse function composition (do f first, then g)
infixr 1 >.>
(f >.> g) x = g (f x)


infixr 1 |>
(|>) :: a -> (a->b) -> b     -- Reverse application
x |> f = f x


infixr 1 |>>
(|>>) :: (TrieMap m2, Functor f)
      => (XT f (m2 a) -> XF f (m1 (m2 a)))
      -> XF f (m2 a)  -> XF f (m1 (m2 a))
(|>>) f g = f (fmap Just . g . deMaybe)


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
  emptyTM  = EmptySEM
  alterTM  = alterSEM
  foldTM   = foldSEM

alterSEM :: (TrieMap tm, Functor f) => TrieKey tm -> XT f v -> SEMap tm v -> f (SEMap tm v)
alterSEM k xt EmptySEM
  = xt Nothing <&> \case
      Nothing -> EmptySEM
      Just v  -> SingleSEM k v
alterSEM k1 xt tm@(SingleSEM k2 v2)
  | k1 == k2 = xt (Just v2) <&> \case
                  Nothing -> EmptySEM
                  Just v' -> SingleSEM k2 v'
  | otherwise = xt Nothing <&> \case
                  Nothing -> tm
                  Just v1  -> MultiSEM $ insertTM k1 v1 (insertTM k2 v2 emptyTM)

alterSEM k xt (MultiSEM tm)
  = MultiSEM <$> alterTM k xt tm


foldSEM :: TrieMap tm => (v -> a -> a) -> SEMap tm v -> a -> a
foldSEM _ EmptySEM        z = z
foldSEM f (SingleSEM _ v) z = f v z
foldSEM f (MultiSEM tm)   z = foldTM f tm z


{- *********************************************************************
*                                                                      *
                  ListMap
*                                                                      *
********************************************************************* -}

type ListMap tm = SEMap (ListMap' tm)

data ListMap' tm a
  = LM { lm_nil  :: Maybe a
       , lm_cons :: tm (ListMap tm a) }

upd_nil :: Functor f => XF f (Maybe a) -> XF f (ListMap' tm a)
upd_nil xt lm@(LM { .. }) = xt lm_nil <&> \nil' -> lm { lm_nil = nil' }

upd_cons :: Functor f => XF f (tm (ListMap tm a)) -> XF f (ListMap' tm a)
upd_cons xt lm@(LM { .. }) = xt lm_cons <&> \cons' -> lm { lm_cons = cons' }

instance TrieMap tm => TrieMap (ListMap' tm) where
   type TrieKey (ListMap' tm) = [TrieKey tm]
   emptyTM  = emptyList
   alterTM  = alterList
   foldTM   = foldList

emptyList :: TrieMap tm => ListMap' tm a
emptyList = LM { lm_nil = Nothing, lm_cons = emptyTM }

alterList :: (TrieMap tm, Functor f) => [TrieKey tm] -> XT f v -> XF f (ListMap' tm v)
alterList []      xt = upd_nil  xt
alterList (k:ks') xt = upd_cons (alterTM k |>> alterTM ks' xt)

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
       , em_lit  :: Map.Map Lit a
       , em_lam  :: ExprMap a
       }

upd_bvar :: Functor f => XF f (BoundVarMap a) -> XF f (ExprMap' a)
upd_bvar xf em@(EM { .. }) = xf em_bvar <&> \bvar' -> em { em_bvar = bvar' }

upd_fvar :: Functor f => XF f (FreeVarMap a) -> XF f (ExprMap' a)
upd_fvar xf em@(EM { .. }) = xf em_fvar <&> \fvar' -> em { em_fvar = fvar' }

upd_app :: Functor f => XF f (ExprMap (ExprMap a)) -> XF f (ExprMap' a)
upd_app xf em@(EM { .. }) = xf em_app <&> \app' -> em { em_app = app' }

upd_lit :: Functor f => XF f (Map.Map Lit a) -> XF f (ExprMap' a)
upd_lit xf em@(EM { .. }) = xf em_lit <&> \lit' -> em { em_lit = lit' }

upd_lam :: Functor f => XF f (ExprMap a) -> XF f (ExprMap' a)
upd_lam xf em@(EM { .. }) = xf em_lam <&> \lam' -> em { em_lam = lam' }

deriving instance (Show (TrieKey ExprMap'), Show v)
               => Show (ExprMap' v)

instance TrieMap ExprMap' where
  type TrieKey ExprMap' = DeBruijn Expr
  emptyTM  = emptyExprMap
  alterTM  = alterExpr
  foldTM   = foldExpr

emptyExprMap :: ExprMap' a
emptyExprMap
  = EM { em_fvar = emptyFVM
       , em_bvar = emptyBVM
       , em_app  = emptyTM
       , em_lit  = Map.empty
       , em_lam  = emptyTM }

alterExpr :: Functor f => DeBruijn Expr -> XT f v -> ExprMap' v -> f (ExprMap' v)
alterExpr (D dbe e) xt
  = case e of
      Var x     -> case lookupDBE x dbe of
                      Just bv -> upd_bvar (alterBoundVarOcc bv xt)
                      Nothing -> upd_fvar (alterFreeVarOcc  x  xt)
      Lit lit   -> upd_lit (Map.alterF xt lit)
      App e1 e2 -> upd_app (alterTM (D dbe e1) |>> alterTM (D dbe e2) xt)
      Lam x e   -> upd_lam (alterTM (D (extendDBE x dbe) e) xt)

foldExpr :: (v -> a -> a) -> ExprMap' v -> a -> a
foldExpr f (EM {..})
  = foldBVM f em_bvar .
    foldFVM f em_fvar .
    foldTM (foldTM f) em_app .
    (\z -> foldr f z em_lit) .
    foldTM f em_lam

-- | For debugging purposes. Draw with 'containers:Data.Tree.drawTree' or
-- 'tree-view:Data.Tree.View.showTree'. The latter uses much less screen space.
exprMapToTree :: Show v => ExprMap v -> Tree String
exprMapToTree = Node "." . go_sem (\v -> [ Node (show v) [] ])
  where
    go_sem go_val EmptySEM = []
    go_sem go_val (SingleSEM k v) = [ Node (show k) (go_val v)]
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
    go_lam  go_val em  = [ Node "λ" (go_sem go_val em) ]
    go_app :: (v -> Forest String) -> ExprMap (ExprMap v) -> Forest String
    go_app  go_val em  = [ Node "@" (go_sem (go_sem go_val) em) ]

{- *********************************************************************
*                                                                      *
                  Matching ExprMap
*                                                                      *
********************************************************************* -}

{-
type MExprMapX = SEMap MExprMapX'

data MExprMapX' a
  = MEM { mem_tvar   :: Maybe a          -- First occurrence of a template tyvar
        , mem_xvar   :: PatOccs a       -- Subsequent occurrence of a template tyvar

        , mem_bvar   :: BoundVarMap a    -- Occurrence of a lam-bound tyvar
        , mem_fvar   :: FreeVarMap a     -- Occurrence of a completely free tyvar

        , mem_fun    :: MExprMapX (MExprMapX a)
        , mem_tycon  :: Map.Map Lit a
        , mem_lam :: MExprMap a
        }

deriving instance (Show (TrieKey MExprMapX'), Show v)
               => Show (MExprMapX' v)

emptyMExprMapX :: MExprMapX' a
emptyMExprMapX
  = MEM { mem_tvar   = Nothing
        , mem_fvar   = emptyFVM
        , mem_xvar   = []
        , mem_bvar   = emptyBVM
        , mem_fun    = emptyTM
        , mem_tycon  = Map.empty
        , mem_lam    = emptyTM }

lkT :: DeBruijn Expr -> (PatSubst, MExprMapX' a) -> [(PatSubst, a)]
lkT (D dbe ty) (tsubst, MEM { .. })
  = tmpl_var_bndr ++ rest
  where
     rest = tmpl_var_occs ++ go ty

     go (Var tv)
       | Just bv <- lookupDBE tv dbe = lkBoundVarOcc bv (tsubst, mem_bvar)
       | otherwise                      = lkFreeVarOcc  tv (tsubst, mem_fvar)
     go (App t1 t2) = concatMap (lkT (D dbe t2)) $
                      lkT (D dbe t1) (tsubst, mem_fun)
     go (Lit tc)    = lkTC tc tsubst mem_tycon

     go (Lam tv ty) = lkT (D (extendDBE tv dbe) ty) (tsubst, mem_lam)

     tmpl_var_bndr | Just x <- mem_tvar
--                   , null rest    -- This one line does overlap!
                   , noCaptured dbe ty
                   = [(extendPatSubst ty tsubst, x)]
                   | otherwise
                   = []

     tmpl_var_occs = [ (tsubst, x)
                     | (tmpl_var, x) <- mem_xvar
                     , deBruijnize (lookupPatSubst tmpl_var tsubst)
                       `eqDBExpr` (D dbe ty)
                     ]

lkTC :: Lit -> PatSubst -> Map.Map Lit a -> [(PatSubst, a)]
lkTC tc tsubst tc_map = case Map.lookup tc tc_map of
                           Nothing -> []
                           Just x  -> [(tsubst,x)]

xtT :: PatVarSet -> DeBruijn Expr
    -> (PatKeys -> XT a)
    -> PatKeys -> MExprMapX' a -> MExprMapX' a
xtT tmpls (D dbe ty) f tkeys m@(MEM {..})
  = go ty
  where
   go (Var tv)
      -- Second or subsequent occurrence of a template tyvar
      | Just xv <- lookupDBE tv tkeys  = m { mem_xvar = xtPatVarOcc xv (f tkeys) mem_xvar }

      -- First occurrence of a template tyvar
      | tv `Set.member` tmpls = m { mem_tvar = f (extendDBE tv tkeys) mem_tvar  }

      -- Occurrence of a lam-bound var
      | Just bv <- lookupDBE tv dbe = m { mem_bvar = xtBoundVarOcc bv (f tkeys) mem_bvar }

      -- A completely free variable
      | otherwise = m { mem_fvar = xtFreeVarOcc  tv (f tkeys) mem_fvar }

   go (Lit tc)  = m { mem_tycon = xtTC tc (f tkeys) mem_tycon }
   go (App t1 t2) = m { mem_fun   = xtT tmpls (D dbe t1)
                                         (liftXT (xtT tmpls (D dbe t2) f))
                                         tkeys mem_fun }
   go (Lam tv ty) = m { mem_lam = xtT tmpls (D (extendDBE tv dbe) ty)
                                             f tkeys mem_lam }


xtTC :: Lit -> XT a -> Map.Map Lit a ->  Map.Map Lit a
xtTC tc f m = Map.alter f tc m

liftXT :: (PatKeys -> MExprMap a -> MExprMap a)
        -> PatKeys -> Maybe (MExprMap a) -> Maybe (MExprMap a)
liftXT insert tkeys Nothing  = Just (insert tkeys emptyMExprMap)
liftXT insert tkeys (Just m) = Just (insert tkeys m)


{- *********************************************************************
*                                                                      *
                  MExprMap
*                                                                      *
********************************************************************* -}

type Match a = ([(PatVar, PatKey)], a)
type MExprMap a = MExprMapX (Match a)

emptyMExprMap :: MExprMap a
emptyMExprMap = emptyMExprMapX

insertMExprMap :: [Var]   -- Pattern variables
               -> Expr    -- Paterrn
               -> a -> MExprMap a -> MExprMap a
insertMExprMap tmpl_tvs ty x tm
  = xtT tmpl_set (deBruijnize ty) f emptyDBE tm
  where
    tmpl_set = Set.fromList tmpl_tvs

    f :: PatKeys -> XT (Match a)
    f tkeys _ = Just (map inst_key tmpl_tvs, x)
     -- The "_" means just overwrite previous value
     where
        inst_key :: Var -> (PatVar, PatKey)
        inst_key tv = case lookupDBE tv tkeys of
                         Nothing  -> error ("Unbound tmpl var " ++ tv)
                         Just key -> (tv, key)

lookupMExprMap :: Expr -> MExprMap a -> [ ([(PatVar,Expr)], a) ]
lookupMExprMap ty tm
  = [ (map (lookup tsubst) prs, x)
    | (tsubst, (prs, x)) <- lkT (deBruijnize ty) (emptyPatSubst, tm) ]
  where
    lookup :: PatSubst -> (PatVar, PatKey) -> (PatVar, Expr)
    lookup tsubst (tv, key) = (tv, lookupPatSubst key tsubst)

-}

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
        guard $ not $ null args
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

instance (Pretty k, Pretty a) => Pretty (Map.Map k a) where
   ppr m = brackets $ commaSep [ ppr k <+> text ":->" <+> ppr v
                               | (k,v) <- Map.toList m ]

instance Pretty a => Pretty (IntMap.IntMap a) where
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

instance (Pretty a) => Pretty (DeBruijn a) where
  ppr (D bv a) = text "D" PP.<> braces (sep [ppr bv, ppr a])

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

instance Pretty PatSubst where
   ppr (TS { ts_subst = subst, ts_next = next })
     = text "TS" PP.<> parens (int next) <+> ppr subst
-}