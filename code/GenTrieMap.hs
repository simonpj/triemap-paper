{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables,
             StandaloneDeriving, UndecidableInstances #-}

module GenTrieMap where

import Data.List( foldl' )
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.IntMap as IntMap
import Data.Kind
import Control.Monad
import Data.Maybe( isJust )
import Text.PrettyPrint as PP
import Debug.Trace
import Data.Char

{- *********************************************************************
*                                                                      *
                   Tests
*                                                                      *
********************************************************************* -}

item1, item2, item3, item4 :: (String, [Var], Expr)
item1 = ("item1", ["a"], read "a -> Int")
item2 = ("item2", ["a"], read "a -> a")
item3 = ("item3", [],    read "Int -> Int")
item4 = ("item4", ["a", "b"], read "b -> a")

ty1, ty2, ty3 :: Expr
ty1 = read "Int -> Int"
ty2 = read "Char -> Char"
ty3 = read "Char -> Int"

ins :: MExprMap String -> (String, [Var], Expr) -> MExprMap String
ins m (s,tvs,ty) = insertMExprMap tvs ty s m

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
deBruijnize ty = D emptyDBE ty

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
noCaptured bv_env ty
  = not (anyFreeVarsOfExpr captured ty)
  where
    captured tv = isJust (lookupDBE tv bv_env)

eqDeBT :: DeBruijn Expr -> DeBruijn Expr -> Bool
eqDeBT (D env1 (App s1 t1)) (D env2 (App s2 t2))
  = eqDeBT (D env1 s1) (D env2 s2) &&
    eqDeBT (D env1 t1) (D env2 t2)

eqDeBT (D env1 (Var tv1)) (D env2 (Var tv2))
  = case (lookupDBE tv1 env1, lookupDBE tv2 env2) of
      (Just bvi1, Just bvi2) -> bvi1 == bvi2
      (Nothing,   Nothing)   -> tv1 == tv2
      _                      -> False

eqDeBT (D _ (Lit tc1)) (D _ (Lit tc2))
  = tc1 == tc2

eqDeBT (D env1 (Lam tv1 t1)) (D env2 (Lam tv2 t2))
  = eqDeBT (D (extendDBE tv1 env1) t1)
           (D (extendDBE tv2 env2) t2)

eqDeBT _ _ = False


{- *********************************************************************
*                                                                      *
                   Template variables
*                                                                      *
********************************************************************* -}

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


{- *********************************************************************
*                                                                      *
               Bound variables
*                                                                      *
********************************************************************* -}

type BoundVar = BoundVarKey  -- Bound variables are deBruijn numbered
type BoundVarMap a = IntMap.IntMap a

emptyBoundVarMap :: BoundVarMap a
emptyBoundVarMap = IntMap.empty

lookupBoundVarMap :: BoundVar -> BoundVarMap a -> Maybe a
lookupBoundVarMap = IntMap.lookup

extendBoundVarMap :: BoundVar -> a -> BoundVarMap a -> BoundVarMap a
extendBoundVarMap = IntMap.insert

lkBoundVarOcc :: BoundVar -> (PatSubst, BoundVarMap a) -> [(PatSubst, a)]
lkBoundVarOcc var (tsubst, env) = case lookupBoundVarMap var env of
                                     Just x  -> [(tsubst,x)]
                                     Nothing -> []

xtBoundVarOcc :: BoundVar -> XT a -> BoundVarMap a -> BoundVarMap a
xtBoundVarOcc tv f tm = IntMap.alter f tv tm


{- *********************************************************************
*                                                                      *
                   Free variables
*                                                                      *
********************************************************************* -}

type FreeVarMap a = Map.Map Var a

emptyFreeVarMap :: FreeVarMap a
emptyFreeVarMap = Map.empty

lookupFreeVarMap :: FreeVarMap a -> Var -> Maybe a
lookupFreeVarMap env tv = Map.lookup tv env

extendFreeVarMap :: FreeVarMap a -> Var -> a -> FreeVarMap a
extendFreeVarMap env tv val = Map.insert tv val env

xtFreeVarOcc :: Var -> XT a -> FreeVarMap a -> FreeVarMap a
xtFreeVarOcc tv f tm = Map.alter f tv tm

lkFreeVarOcc :: Var -> (PatSubst, FreeVarMap a) -> [(PatSubst, a)]
lkFreeVarOcc var (tsubst, env) = case Map.lookup var env of
                                    Just x  -> [(tsubst,x)]
                                    Nothing -> []


{- *********************************************************************
*                                                                      *
                  The TrieMap class
*                                                                      *
********************************************************************* -}

class TrieMap m where
   type TrieKey m :: Type
   emptyTM  :: m a
   lookupTM :: forall b. TrieKey m -> m b -> Maybe b
   alterTM  :: forall b. TrieKey m -> XT b -> m b -> m b
   foldTM   :: (a -> b -> b) -> m a -> b -> b

--   mapTM    :: (a->b) -> m a -> m b
--   filterTM :: (a -> Bool) -> m a -> m a
--   unionTM  ::  m a -> m a -> m a

type XT a = Maybe a -> Maybe a  -- How to alter a non-existent elt (Nothing)
                                --               or an existing elt (Just)

-- Recall that
--   Control.Monad.(>=>) :: (a -> Maybe b) -> (b -> Maybe c) -> a -> Maybe c

(>.>) :: (a -> b) -> (b -> c) -> a -> c
-- Reverse function composition (do f first, then g)
infixr 1 >.>
(f >.> g) x = g (f x)


(|>) :: a -> (a->b) -> b     -- Reverse application
infixr 1 |>
x |> f = f x

(|>>) :: TrieMap m2 => (XT (m2 a) -> m1 (m2 a) -> m1 (m2 a))
                    -> (m2 a -> m2 a)
                    -> m1 (m2 a) -> m1 (m2 a)
infixr 1 |>>
(|>>) f g = f (Just . g . deMaybe)

deMaybe :: TrieMap m => Maybe (m a) -> m a
deMaybe Nothing  = emptyTM
deMaybe (Just m) = m


{- *********************************************************************
*                                                                      *
                  ListMap
*                                                                      *
********************************************************************* -}

data ListMap tm a
  = EmptyLM
  | LM { lm_nil  :: Maybe a
       , lm_cons :: tm (ListMap tm a) }

instance TrieMap tm => TrieMap (ListMap tm) where
   type TrieKey (ListMap tm) = [TrieKey tm]
   emptyTM  = emptyListMap
   lookupTM = lookupListMap

emptyListMap :: ListMap m a
emptyListMap = EmptyLM

lookupListMap :: TrieMap tm => [TrieKey tm] -> ListMap tm a -> Maybe a
lookupListMap _   EmptyLM = Nothing
lookupListMap key (LM { .. })
  = case key of
      []     -> lm_nil
      (k:ks) -> lm_cons |> lookupTM k >=> lookupListMap ks


{- *********************************************************************
*                                                                      *
                  Singleton and empty map
*                                                                      *
********************************************************************* -}

data SEMap tm k v
  = EmptySEM
  | SingleSEM k v
  | MultiSEM  tm
  deriving( Show )

lookupSEM :: Eq k => (k -> tm           -> Maybe v)
                   -> k -> SEMap tm k v -> Maybe v
lookupSEM _ _  EmptySEM = Nothing
lookupSEM _ tk (SingleSEM pk v) | tk == pk  = Just v
                                | otherwise = Nothing
lookupSEM lookup target_key sem
  = case sem of
      EmptySEM    -> Nothing
      MultiSEM tm -> lookup target_key tm
      SingleSEM pat_key val
         | target_key == pat_key -> Just val
         | otherwise             -> Nothing


{- *********************************************************************
*                                                                      *
                  ExprMap
*                                                                      *
********************************************************************* -}

data ExprMap a
  = EmptyEM
  | EM { tm_tvar   :: Maybe a          -- First occurrence of a template tyvar
       , tm_xvar   :: PatOccs a       -- Subsequent occurrence of a template tyvar

       , tm_bvar   :: BoundVarMap a    -- Occurrence of a forall-bound tyvar
       , tm_fvar   :: FreeVarMap a     -- Occurrence of a completely free tyvar

       , tm_fun    :: ExprMap (ExprMap a)
       , tm_tycon  :: Map.Map Lit a
       , tm_forall :: ExprMap a
       }
  deriving( Show )

emptyExprMap :: ExprMap a
emptyExprMap = EmptyEM

mkEmptyExprMap :: ExprMap a
mkEmptyExprMap
  = EM { tm_tvar   = Nothing
       , tm_fvar   = emptyFreeVarMap
       , tm_xvar   = []
       , tm_bvar   = emptyBoundVarMap
       , tm_fun    = emptyExprMap
       , tm_tycon  = Map.empty
       , tm_forall = emptyExprMap }

lkT :: DeBruijn Expr -> (PatSubst, ExprMap a) -> [(PatSubst, a)]
lkT _ (_, EmptyEM)
  = []
lkT (D bv_env ty) (tsubst, EM { .. })
  = tmpl_var_bndr ++ rest
  where
     rest = tmpl_var_occs ++ go ty

     go (Var tv)
       | Just bv <- lookupDBE tv bv_env = lkBoundVarOcc bv (tsubst, tm_bvar)
       | otherwise                      = lkFreeVarOcc  tv (tsubst, tm_fvar)
     go (App t1 t2) = concatMap (lkT (D bv_env t2)) $
                      lkT (D bv_env t1) (tsubst, tm_fun)
     go (Lit tc)    = lkTC tc tsubst tm_tycon

     go (Lam tv ty) = lkT (D (extendDBE tv bv_env) ty) (tsubst, tm_forall)

     tmpl_var_bndr | Just x <- tm_tvar
--                   , null rest    -- This one line does overlap!
                   , noCaptured bv_env ty
                   = [(extendPatSubst ty tsubst, x)]
                   | otherwise
                   = []

     tmpl_var_occs = [ (tsubst, x)
                     | (tmpl_var, x) <- tm_xvar
                     , deBruijnize (lookupPatSubst tmpl_var tsubst)
                       `eqDeBT` (D bv_env ty)
                     ]

lkTC :: Lit -> PatSubst -> Map.Map Lit a -> [(PatSubst, a)]
lkTC tc tsubst tc_map = case Map.lookup tc tc_map of
                           Nothing -> []
                           Just x  -> [(tsubst,x)]

xtT :: PatVarSet -> DeBruijn Expr
    -> (PatKeys -> XT a)
    -> PatKeys -> ExprMap a -> ExprMap a
xtT tmpls ty f tkeys EmptyEM
 = xtT tmpls ty f tkeys mkEmptyExprMap
xtT tmpls (D bv_env ty) f tkeys m@(EM {..})
  = go ty
  where
   go (Var tv)
      -- Second or subsequent occurrence of a template tyvar
      | Just xv <- lookupDBE tv tkeys  = m { tm_xvar = xtPatVarOcc xv (f tkeys) tm_xvar }

      -- First occurrence of a template tyvar
      | tv `Set.member` tmpls = m { tm_tvar = f (extendDBE tv tkeys) tm_tvar  }

      -- Occurrence of a forall-bound var
      | Just bv <- lookupDBE tv bv_env = m { tm_bvar = xtBoundVarOcc bv (f tkeys) tm_bvar }

      -- A completely free variable
      | otherwise = m { tm_fvar = xtFreeVarOcc  tv (f tkeys) tm_fvar }

   go (Lit tc)  = m { tm_tycon = xtTC tc (f tkeys) tm_tycon }
   go (App t1 t2) = m { tm_fun   = xtT tmpls (D bv_env t1)
                                         (liftXT (xtT tmpls (D bv_env t2) f))
                                         tkeys tm_fun }
   go (Lam tv ty) = m { tm_forall = xtT tmpls (D (extendDBE tv bv_env) ty)
                                             f tkeys tm_forall }


xtTC :: Lit -> XT a -> Map.Map Lit a ->  Map.Map Lit a
xtTC tc f m = Map.alter f tc m

liftXT :: (PatKeys -> ExprMap a -> ExprMap a)
        -> PatKeys -> Maybe (ExprMap a) -> Maybe (ExprMap a)
liftXT insert tkeys Nothing  = Just (insert tkeys emptyExprMap)
liftXT insert tkeys (Just m) = Just (insert tkeys m)


{- *********************************************************************
*                                                                      *
                  ExprMap
*                                                                      *
********************************************************************* -}

type Match a = ([(PatVar, PatKey)], a)
type MExprMap a = ExprMap (Match a)

emptyMExprMap :: MExprMap a
emptyMExprMap = emptyExprMap

insertMExprMap :: forall a. [Var]   -- Template type variables
                        -> Expr      -- Template
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

lookupExprMap :: Expr -> MExprMap a -> [ ([(PatVar,Expr)], a) ]
lookupExprMap ty tm
  = [ (map (lookup tsubst) prs, x)
    | (tsubst, (prs, x)) <- lkT (deBruijnize ty) (emptyPatSubst, tm) ]
  where
    lookup :: PatSubst -> (PatVar, PatKey) -> (PatVar, Expr)
    lookup tsubst (tv, key) = (tv, lookupPatSubst key tsubst)


{- *********************************************************************
*                                                                      *
                   Pretty-printing
*                                                                      *
********************************************************************* -}

piPrec :: Int
piPrec = 1

-- | Example output: @Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b@
instance Show Expr where
  showsPrec _ (Var v)       = showString v
  showsPrec _ (Lit tc)      = showString tc
  showsPrec p (App arg res)   = showParen (p > piPrec) $
    showsPrec (piPrec+1) arg . showString " -> " . showsPrec piPrec res
  showsPrec p (Lam v body) = showParen (p > piPrec) $
    showString "∀" . showString v . showString ". " . showsPrec piPrec body

-- | This monster parses Exprs in the REPL etc. Accepts syntax like
-- @Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b@
--
-- >>> read "Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b" :: Expr
-- Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b
instance Read Expr where
  readsPrec p s = readParen False (\s -> do
                    (v@(_:_), r) <- lex s
                    guard (all isAlphaNum v)
                    pure $ if isLower (head v)
                      then (Var v, r)
                      else (Lit v, r)) s
                  ++
                  readParen (p > piPrec) (\s -> do
                    (tok, r1) <- lex s
                    case tok of
                      [c] | c `elem` "∀@#%" -> do -- multiple short-hands for Lam
                        (Var v, r2) <- readsPrec (piPrec+1) r1
                        (".", r3) <- lex r2
                        (body, r4) <- readsPrec piPrec r3
                        pure (Lam v body, r4)
                      _ -> do -- App
                        (arg, r1) <- readsPrec (piPrec+1) s
                        ("->", r2) <- lex r1
                        (res, r3) <- readsPrec piPrec r2
                        pure (App arg res, r3)) s

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

instance Pretty a => Pretty (ExprMap a) where
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "tm_tvar =" <+> ppr tm_tvar
                    , text "tm_xvar =" <+> ppr tm_xvar
                    , text "tm_bvar =" <+> ppr tm_bvar
                    , text "tm_fvar =" <+> ppr tm_fvar
                    , text "tm_fun =" <+> ppr tm_fun
                    , text "tm_tycon =" <+> ppr tm_tycon
                    , text "tm_forall =" <+> ppr tm_forall ])

instance Pretty PatSubst where
   ppr (TS { ts_subst = subst, ts_next = next })
     = text "TS" PP.<> parens (int next) <+> ppr subst
