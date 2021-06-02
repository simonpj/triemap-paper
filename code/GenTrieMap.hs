{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables #-}

module GenTrieMap where

import Data.List( foldl' )
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.IntMap as IntMap
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

item1, item2, item3, item4 :: (String, [TyVar], Type)
item1 = ("item1", ["a"], read "a -> Int")
item2 = ("item2", ["a"], read "a -> a")
item3 = ("item3", [],    read "Int -> Int")
item4 = ("item4", ["a", "b"], read "b -> a")

ty1, ty2, ty3 :: Type
ty1 = read "Int -> Int"
ty2 = read "Char -> Char"
ty3 = read "Char -> Int"

ins :: TypeMap String -> (String, [TyVar], Type) -> TypeMap String
ins m (s,tvs,ty) = insertTypeMap tvs ty s m

{- *********************************************************************
*                                                                      *
                   Type
*                                                                      *
********************************************************************* -}

type TyVar    = String
type TyCon    = String

data Type = TyVarTy TyVar
          | FunTy Type Type
          | ForAllTy TyVar Type
          | TyConTy TyCon

anyFreeVarsOfType :: (TyVar -> Bool) -> Type -> Bool
-- True if 'p' returns True of any free variable
-- of the type; False otherwise
anyFreeVarsOfType p ty
  = go Set.empty ty
  where
    go bvs (TyVarTy tv) | tv `Set.member` bvs = False
                        | otherwise           = p tv
    go bvs (FunTy t1 t2)    = go bvs t1 || go bvs t2
    go bvs (ForAllTy tv ty) = go (Set.insert tv bvs) ty
    go _   (TyConTy _)      = False

{- *********************************************************************
*                                                                      *
                   DeBruijn
*                                                                      *
********************************************************************* -}

-- | @DeBruijn a@ represents @a@ modulo alpha-renaming.  This is achieved
-- by equipping the value with a 'CmEnv', which tracks an on-the-fly deBruijn
-- numbering.  This allows us to define an 'Eq' instance for @DeBruijn a@, even
-- if this was not (easily) possible for @a@.  Note: we purposely don't
-- export the constructor.  Make a helper function if you find yourself
-- needing it.
data DeBruijn a = D CmEnv a

-- | Synthesizes a @DeBruijn a@ from an @a@, by assuming that there are no
-- bound binders (an empty 'CmEnv').  This is usually what you want if there
-- isn't already a 'CmEnv' in scope.
deBruijnize :: a -> DeBruijn a
deBruijnize ty = D emptyCME ty

instance Eq (DeBruijn a) => Eq (DeBruijn [a]) where
    D _   []     == D _    []       = True
    D env (x:xs) == D env' (x':xs') = D env x  == D env' x' &&
                                      D env xs == D env' xs'
    _            == _               = False

instance Eq (DeBruijn a) => Eq (DeBruijn (Maybe a)) where
    D _   Nothing  == D _    Nothing   = True
    D env (Just x) == D env' (Just x') = D env x  == D env' x'
    _              == _                = False

noCaptured :: CmEnv -> Type -> Bool
-- True iff no free var of the type is bound by CmEnv
noCaptured bv_env ty
  = not (anyFreeVarsOfType captured ty)
  where
    captured tv = isJust (lookupCME tv bv_env)

eqDeBT :: DeBruijn Type -> DeBruijn Type -> Bool
eqDeBT (D env1 (FunTy s1 t1)) (D env2 (FunTy s2 t2))
  = eqDeBT (D env1 s1) (D env2 s2) &&
    eqDeBT (D env1 t1) (D env2 t2)

eqDeBT (D env1 (TyVarTy tv1)) (D env2 (TyVarTy tv2))
  = case (lookupCME tv1 env1, lookupCME tv2 env2) of
      (Just bvi1, Just bvi2) -> bvi1 == bvi2
      (Nothing,   Nothing)   -> tv1 == tv2
      _                      -> False

eqDeBT (D _ (TyConTy tc1)) (D _ (TyConTy tc2))
  = tc1 == tc2

eqDeBT (D env1 (ForAllTy tv1 t1)) (D env2 (ForAllTy tv2 t2))
  = eqDeBT (D (extendCME tv1 env1) t1)
           (D (extendCME tv2 env2) t2)

eqDeBT _ _ = False

{- *********************************************************************
*                                                                      *
                  TypeMap
*                                                                      *
********************************************************************* -}

type Match a = ([(TmplVar, TmplKey)], a)
type TypeMap a = GTypeMap (Match a)

emptyTypeMap :: TypeMap a
emptyTypeMap = emptyGTypeMap

insertTypeMap :: forall a. [TyVar]   -- Template type variables
                        -> Type      -- Template
                        -> a -> TypeMap a -> TypeMap a
insertTypeMap tmpl_tvs ty x tm
  = xtT tmpl_set (deBruijnize ty) f emptyCME tm
  where
    tmpl_set = Set.fromList tmpl_tvs

    f :: TmplKeys -> XT (Match a)
    f tkeys _ = Just (map inst_key tmpl_tvs, x)
     -- The "_" means just overwrite previous value
     where
        inst_key :: TyVar -> (TmplVar, TmplKey)
        inst_key tv = case lookupCME tv tkeys of
                         Nothing  -> error ("Unbound tmpl var " ++ tv)
                         Just key -> (tv, key)

lookupTypeMap :: Type -> TypeMap a -> [ ([(TmplVar,Type)], a) ]
lookupTypeMap ty tm
  = [ (map (lookup tsubst) prs, x)
    | (tsubst, (prs, x)) <- lkT (deBruijnize ty) (emptyTmplSubst, tm) ]
  where
    lookup :: TmplSubst -> (TmplVar, TmplKey) -> (TmplVar, Type)
    lookup tsubst (tv, key) = (tv, lookupTmplSubst key tsubst)


{- *********************************************************************
*                                                                      *
                  GTypeMap
*                                                                      *
********************************************************************* -}

data GTypeMap a
  = EmptyTM
  | TM { tm_tvar   :: Maybe a          -- First occurrence of a template tyvar
       , tm_xvar   :: TmplOccs a       -- Subsequent occurrence of a template tyvar

       , tm_bvar   :: BoundVarMap a    -- Occurrence of a forall-bound tyvar
       , tm_fvar   :: FreeVarMap a     -- Occurrence of a completely free tyvar

       , tm_fun    :: GTypeMap (GTypeMap a)
       , tm_tycon  :: Map.Map TyCon a
       , tm_forall :: GTypeMap a
       }
  deriving( Show )

emptyGTypeMap :: GTypeMap a
emptyGTypeMap = EmptyTM

mkEmptyGTypeMap :: GTypeMap a
mkEmptyGTypeMap
  = TM { tm_tvar   = Nothing
       , tm_fvar   = emptyFreeVarMap
       , tm_xvar   = []
       , tm_bvar   = emptyBoundVarMap
       , tm_fun    = emptyGTypeMap
       , tm_tycon  = Map.empty
       , tm_forall = emptyGTypeMap }

lkT :: DeBruijn Type -> (TmplSubst, GTypeMap a) -> [(TmplSubst, a)]
-- lk = lookup
lkT _ (_, EmptyTM)
  = []
lkT (D bv_env ty) (tsubst, TM { .. })
  = tmpl_var_bndr ++ rest
  where
     rest = tmpl_var_occs ++ go ty

     go (TyVarTy tv)
       | Just bv <- lookupCME tv bv_env = lkBoundVarOcc bv (tsubst, tm_bvar)
       | otherwise                      = lkFreeVarOcc  tv (tsubst, tm_fvar)
     go (FunTy t1 t2)    = concatMap (lkT (D bv_env t2)) $
                           lkT (D bv_env t1) (tsubst, tm_fun)
     go (TyConTy tc)     = lkTC tc (tsubst, tm_tycon)

     go (ForAllTy tv ty) = lkT (D (extendCME tv bv_env) ty) (tsubst, tm_forall)

     tmpl_var_bndr | Just x <- tm_tvar
--                   , null rest    -- This one line does overlap!
                   , noCaptured bv_env ty
                   = [(extendTmplSubst ty tsubst, x)]
                   | otherwise
                   = []

     tmpl_var_occs = [ (tsubst, x)
                     | (tmpl_var, x) <- tm_xvar
                     , deBruijnize (lookupTmplSubst tmpl_var tsubst)
                       `eqDeBT` (D bv_env ty)
                     ]

lkTC :: TyCon -> (TmplSubst, Map.Map TyCon a) -> [(TmplSubst, a)]
lkTC tc (tsubst, tc_map) = case Map.lookup tc tc_map of
                             Nothing -> []
                             Just x  -> [(tsubst,x)]

xtT :: TmplVarSet -> DeBruijn Type
    -> (TmplKeys -> XT a)
    -> TmplKeys -> GTypeMap a -> GTypeMap a
-- xt = alter
xtT tmpls ty f tkeys EmptyTM
 = xtT tmpls ty f tkeys mkEmptyGTypeMap

xtT tmpls (D bv_env ty) f tkeys m@(TM {..})
  = go ty
  where
   go (TyVarTy tv)
      -- Second or subsequent occurrence of a template tyvar
      | Just xv <- lookupCME tv tkeys  = m { tm_xvar = xtTmplVarOcc xv (f tkeys) tm_xvar }

      -- First occurrence of a template tyvar
      | tv `Set.member` tmpls = m { tm_tvar = f (extendCME tv tkeys) tm_tvar  }

      -- Occurrence of a forall-bound var
      | Just bv <- lookupCME tv bv_env = m { tm_bvar = xtBoundVarOcc bv (f tkeys) tm_bvar }

      -- A completely free variable
      | otherwise = m { tm_fvar = xtFreeVarOcc  tv (f tkeys) tm_fvar }

   go (TyConTy tc)  = m { tm_tycon = xtTC tc (f tkeys) tm_tycon }
   go (FunTy t1 t2) = m { tm_fun   = xtT tmpls (D bv_env t1)
                                         (liftXT (xtT tmpls (D bv_env t2) f))
                                         tkeys tm_fun }
   go (ForAllTy tv ty) = m { tm_forall = xtT tmpls (D (extendCME tv bv_env) ty)
                                             f tkeys tm_forall }


xtTC :: TyCon -> XT a -> Map.Map TyCon a ->  Map.Map TyCon a
xtTC tc f m = Map.alter f tc m

liftXT :: (TmplKeys -> GTypeMap a -> GTypeMap a)
        -> TmplKeys -> Maybe (GTypeMap a) -> Maybe (GTypeMap a)
liftXT insert tkeys Nothing  = Just (insert tkeys emptyGTypeMap)
liftXT insert tkeys (Just m) = Just (insert tkeys m)


{- *********************************************************************
*                                                                      *
                   Template variables
*                                                                      *
********************************************************************* -}

type TmplVar = TyVar
type TmplVarSet = Set.Set TyVar
type TmplKey = CmKey
type TmplKeys = CmEnv  -- Maps TmplVar :-> TmplKey

type TmplOccs a = [(TmplKey,a)]

xtTmplVarOcc :: TmplKey -> XT a -> TmplOccs a -> TmplOccs a
xtTmplVarOcc key f []
  = xtCons key (f Nothing) []
xtTmplVarOcc key f ((key1,x):prs)
  | key == key1 = xtCons key (f (Just x)) prs
  | otherwise   = (key1,x) : xtTmplVarOcc key f prs

xtCons :: TmplKey -> Maybe a -> TmplOccs a -> TmplOccs a
xtCons _   Nothing  tmpl_occs = tmpl_occs
xtCons key (Just x) tmpl_occs = (key,x) : tmpl_occs

---------------
data TmplSubst = TS { ts_subst :: IntMap.IntMap Type     -- Maps TmplKey -> Type
                    , ts_next  :: TmplKey }

emptyTmplSubst :: TmplSubst
emptyTmplSubst = TS { ts_subst = IntMap.empty
                    , ts_next  = 0 }

lookupTmplSubst :: TmplKey -> TmplSubst -> Type
lookupTmplSubst key (TS { ts_subst = subst })
  = case IntMap.lookup key subst of
      Just ty -> ty
      Nothing -> error ("lookupTmplSubst " ++ show key)

extendTmplSubst :: Type -> TmplSubst -> TmplSubst
extendTmplSubst ty (TS { ts_subst = subst, ts_next = n })
  = TS { ts_subst = IntMap.insert n ty subst
       , ts_next  = n+1 }


{- *********************************************************************
*                                                                      *
               Bound variables
*                                                                      *
********************************************************************* -}

type BoundVar = CmKey  -- Bound variables are deBruijn numbered
type BoundVarMap a = IntMap.IntMap a

emptyBoundVarMap :: BoundVarMap a
emptyBoundVarMap = IntMap.empty

lookupBoundVarMap :: BoundVar -> BoundVarMap a -> Maybe a
lookupBoundVarMap = IntMap.lookup

extendBoundVarMap :: BoundVar -> a -> BoundVarMap a -> BoundVarMap a
extendBoundVarMap = IntMap.insert

lkBoundVarOcc :: BoundVar -> (TmplSubst, BoundVarMap a) -> [(TmplSubst, a)]
lkBoundVarOcc var (tsubst, env) = case lookupBoundVarMap var env of
                                     Just x  -> [(tsubst,x)]
                                     Nothing -> []

xtBoundVarOcc :: BoundVar -> XT a -> BoundVarMap a -> BoundVarMap a
xtBoundVarOcc tv f tm = IntMap.alter f tv tm



{- *********************************************************************
*                                                                      *
               Bound variabes and de Bruijn
*                                                                      *
********************************************************************* -}

type CmKey = Int
data CmEnv = CME { cme_next :: !CmKey
                 , cme_env  :: Map.Map TyVar CmKey }

emptyCME :: CmEnv
emptyCME = CME { cme_next = 0, cme_env = Map.empty }

extendCME :: TyVar -> CmEnv -> CmEnv
extendCME tv (CME { cme_next = bv, cme_env = env })
  = CME { cme_next = bv+1, cme_env = Map.insert tv bv env }

lookupCME :: TyVar -> CmEnv -> Maybe BoundVar
lookupCME v (CME { cme_env = env }) = Map.lookup v env


{- *********************************************************************
*                                                                      *
                   Free variables
*                                                                      *
********************************************************************* -}

type FreeVarMap a = Map.Map TyVar a

emptyFreeVarMap :: FreeVarMap a
emptyFreeVarMap = Map.empty

lookupFreeVarMap :: FreeVarMap a -> TyVar -> Maybe a
lookupFreeVarMap env tv = Map.lookup tv env

extendFreeVarMap :: FreeVarMap a -> TyVar -> a -> FreeVarMap a
extendFreeVarMap env tv val = Map.insert tv val env

xtFreeVarOcc :: TyVar -> XT a -> FreeVarMap a -> FreeVarMap a
xtFreeVarOcc tv f tm = Map.alter f tv tm

lkFreeVarOcc :: TyVar -> (TmplSubst, FreeVarMap a) -> [(TmplSubst, a)]
lkFreeVarOcc var (tsubst, env) = case Map.lookup var env of
                                    Just x  -> [(tsubst,x)]
                                    Nothing -> []


{- *********************************************************************
*                                                                      *
                   Generic stuff
*                                                                      *
********************************************************************* -}

type XT a = Maybe a -> Maybe a  -- How to alter a non-existent elt (Nothing)
                                --               or an existing elt (Just)

class TrieMap m where
   type Key m
   emptyTM  :: m a
   lookupTM :: forall b. Key m -> m b -> Maybe b
   alterTM  :: forall b. Key m -> XT b -> m b -> m b

instance TrieMap IntMap.IntMap where
  type Key IntMap.IntMap = Int
  emptyTM       = IntMap.empty
  lookupTM      = IntMap.lookup
  alterTM k f m = IntMap.alter f k m

instance Ord k => TrieMap (Map.Map k) where
  type Key (Map.Map k) = k
  emptyTM       = Map.empty
  lookupTM      = Map.lookup
  alterTM k f m = Map.alter f k m

{-
instance TrieMap TypeMap where
  type Key TypeMap = DeBruijn Type
  emptyTM = emptyTypeMap
  lookupTM = lkT
  alterTM  = xtT
-}

{-
instance TrieMap VarMap where
   type Key VarMap = TyVar
   emptyTM  = emptyVarMap
   lookupTM = lkVar emptyCME
   alterTM  = xtVar emptyCME
-}


-- Recall that
--   Control.Monad.(>=>) :: (a -> Maybe b) -> (b -> Maybe c) -> a -> Maybe c

(>.>) :: (a -> b) -> (b -> c) -> a -> c
-- Reverse function composition (do f first, then g)
infixr 1 >.>
(f >.> g) x = g (f x)


(|>) :: a -> (a->b) -> b     -- Reverse application
infixr 1 |>
x |> f = f x

----------------------
(|>>) :: TrieMap m2
      => (XT (m2 a) -> m1 (m2 a) -> m1 (m2 a))
      -> (m2 a -> m2 a)
      -> m1 (m2 a) -> m1 (m2 a)
infixr 1 |>>
(|>>) f g = f (Just . g . deMaybe)

deMaybe :: TrieMap m => Maybe (m a) -> m a
deMaybe Nothing  = emptyTM
deMaybe (Just m) = m


{- *********************************************************************
*                                                                      *
                   Pretty-printing
*                                                                      *
********************************************************************* -}

piPrec :: Int
piPrec = 1

-- | Example output: @Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b@
instance Show Type where
  showsPrec _ (TyVarTy v)       = showString v
  showsPrec _ (TyConTy tc)      = showString tc
  showsPrec p (FunTy arg res)   = showParen (p > piPrec) $
    showsPrec (piPrec+1) arg . showString " -> " . showsPrec piPrec res
  showsPrec p (ForAllTy v body) = showParen (p > piPrec) $
    showString "∀" . showString v . showString ". " . showsPrec piPrec body

-- | This monster parses Types in the REPL etc. Accepts syntax like
-- @Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b@
--
-- >>> read "Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b" :: Type
-- Bool -> (∀a. Int) -> (Int -> Int) -> ∀b. Char -> b
instance Read Type where
  readsPrec p s = readParen False (\s -> do
                    (v@(_:_), r) <- lex s
                    guard (all isAlphaNum v)
                    pure $ if isLower (head v)
                      then (TyVarTy v, r)
                      else (TyConTy v, r)) s
                  ++
                  readParen (p > piPrec) (\s -> do
                    (tok, r1) <- lex s
                    case tok of
                      [c] | c `elem` "∀@#%" -> do -- multiple short-hands for ForAllTy
                        (TyVarTy v, r2) <- readsPrec (piPrec+1) r1
                        (".", r3) <- lex r2
                        (body, r4) <- readsPrec piPrec r3
                        pure (ForAllTy v body, r4)
                      _ -> do -- FunTy
                        (arg, r1) <- readsPrec (piPrec+1) s
                        ("->", r2) <- lex r1
                        (res, r3) <- readsPrec piPrec r2
                        pure (FunTy arg res, r3)) s

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

instance Pretty Type where
   ppr ty = text (show ty)

instance Pretty a => Pretty (GTypeMap a) where
  ppr EmptyTM = text "EmptyTM"
  ppr (TM {..}) = text "TM" <+> braces (vcat
                    [ text "tm_tvar =" <+> ppr tm_tvar
                    , text "tm_xvar =" <+> ppr tm_xvar
                    , text "tm_bvar =" <+> ppr tm_bvar
                    , text "tm_fvar =" <+> ppr tm_fvar
                    , text "tm_fun =" <+> ppr tm_fun
                    , text "tm_tycon =" <+> ppr tm_tycon
                    , text "tm_forall =" <+> ppr tm_forall ])

instance Pretty TmplSubst where
   ppr (TS { ts_subst = subst, ts_next = next })
     = text "TS" PP.<> parens (int next) <+> ppr subst
