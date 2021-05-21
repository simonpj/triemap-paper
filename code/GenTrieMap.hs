{-# LANGUAGE TypeFamilies, RankNTypes, FlexibleInstances, FlexibleContexts,
             RecordWildCards, ScopedTypeVariables, StandaloneKindSignatures,
             DataKinds, GADTs, TypeApplications, QuantifiedConstraints,
             StandaloneDeriving, DeriveFoldable, TypeOperators, PolyKinds,
             DeriveFunctor #-}

module GenTrieMap where

import qualified Data.Kind as Kind
import Data.Kind ( Constraint )

import Data.List( foldl' )
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.IntMap as IntMap
import Control.Monad
import Data.Maybe( isJust )
import Text.PrettyPrint as PP
import Debug.Trace
import Data.Functor.Const
import Data.Coerce
import Data.Type.Equality
import Unsafe.Coerce

{- *********************************************************************
*                                                                      *
                   Preliminaries
*                                                                      *
********************************************************************* -}

type Ty = Kind.Type

{- *********************************************************************
*                                                                      *
                   Natural numbers
*                                                                      *
********************************************************************* -}

data Nat = Zero | Succ Nat    -- used only at compile time

type SNat :: Nat -> Ty
newtype SNat n where
  UnsafeMkSNat :: { snatToInt :: Int } -> SNat n

type SNatI :: Nat -> Constraint
class SNatI n where
  snat :: SNat n
instance SNatI Zero where
  {-# INLINE snat #-}
  snat = UnsafeMkSNat 0
instance SNatI n => SNatI (Succ n) where
  {-# INLINE snat #-}
  snat = UnsafeMkSNat (1 + snatToInt (snat @n))

unsafeNatEqualityProof :: m :~: n
unsafeNatEqualityProof = unsafeCoerce (Refl @())

eqSNat :: SNat m -> SNat n -> Maybe (m :~: n)
eqSNat (UnsafeMkSNat m) (UnsafeMkSNat n)
  | m == n    = Just unsafeNatEqualityProof
  | otherwise = Nothing

type Fin :: Nat -> Ty     -- numbers in the range [0, n)
newtype Fin n where
  UnsafeMkFin :: { finToInt :: Int } -> Fin n
  deriving (Show, Eq)

maxFin :: forall n. SNatI n => Fin (Succ n)
maxFin = UnsafeMkFin (snatToInt (snat @n))

bumpFinIndex :: Fin n -> Fin (Succ n)
bumpFinIndex = coerce

{- *********************************************************************
*                                                                      *
                   Maps
*                                                                      *
********************************************************************* -}

type FinMap :: Nat -> Ty -> Ty
newtype FinMap n a where
  UnsafeMkFinMap :: IntMap.IntMap a -> FinMap n a

emptyFinMap :: FinMap n a
emptyFinMap = UnsafeMkFinMap IntMap.empty

lookupFinMap :: Fin n -> FinMap n a -> Maybe a
lookupFinMap (UnsafeMkFin k) (UnsafeMkFinMap m) = IntMap.lookup k m

insertFinMap :: Fin n -> a -> FinMap n a -> FinMap n a
insertFinMap (UnsafeMkFin k) val (UnsafeMkFinMap m) = UnsafeMkFinMap (IntMap.insert k val m)

bumpFinMapIndex :: FinMap n a -> FinMap (Succ n) a
bumpFinMapIndex = coerce

{- *********************************************************************
*                                                                      *
                   Vectors
*                                                                      *
********************************************************************* -}

type Vec :: Nat -> Ty -> Ty
data Vec n a where
  Nil  :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a
infixr 5 :>

deriving instance Functor (Vec n)
deriving instance Foldable (Vec n)

{- *********************************************************************
*                                                                      *
                   Tests
*                                                                      *
********************************************************************* -}

item1, item2, item3, item4 :: (String, [TyVar], Type)
item1 = ("item1", ["a"], FunTy (TyVarTy "a") (TyConTy "Int"))
item2 = ("item2", ["a"], FunTy (TyVarTy "a") (TyVarTy "a"))
item3 = ("item3", [],    FunTy (TyConTy "Int") (TyConTy "Int"))
item4 = ("item4", ["a", "b"], FunTy (TyVarTy "b") (TyVarTy "a"))

ty1, ty2, ty3 :: Type
ty1 = FunTy (TyConTy "Int") (TyConTy "Int")
ty2 = FunTy (TyConTy "Char") (TyConTy "Char")
ty3 = FunTy (TyConTy "Char") (TyConTy "Int")

ins :: TypeMap (Const String) -> (String, Vec n TyVar, Type) -> TypeMap (Const String)
ins m (s,tvs,ty) = insertTypeMap tvs ty (Const s) m

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
  deriving( Show )

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

type Match :: (Nat -> Ty) -> Nat -> Ty
data Match a n = MkMatch (Vec n (TmplVar, TmplKey n)) (a n)

type TypeMap :: (Nat -> Ty) -> Ty
type TypeMap a = GTypeMap (Match a) Zero

emptyTypeMap :: TypeMap a
emptyTypeMap = emptyGTypeMap

insertTypeMap :: forall a n. Vec n TyVar   -- Template type variables
                          -> Type          -- Template
                          -> a n -> TypeMap a -> TypeMap a
insertTypeMap tmpl_tvs ty x tm
  = xtT tmpl_set (deBruijnize ty) f emptyTmplKeys tm
  where
    tmpl_set = foldMap Set.singleton tmpl_tvs  -- is there a better way to do this?

    f :: forall m. TmplKeys m -> XT (Match a m)
    f tkeys _ = case unsafeNatEqualityProof @n @m of   -- if this assumption is wrong, an inst_key will fail
     -- The "_" means just overwrite previous value
                  Refl -> Just (MkMatch (fmap inst_key tmpl_tvs) x)

     where
        inst_key :: forall. TyVar -> (TmplVar, TmplKey m)  -- `m` is from outer scope
        inst_key tv = case Map.lookup tv tkeys of
                         Nothing  -> error ("Unbound tmpl var " ++ tv)
                         Just key -> (tv, key)

type LookupTypeMapResult :: (Nat -> Ty) -> Ty
data LookupTypeMapResult a where
  LTMR :: forall n a. Vec n (TmplVar, Type) -> a n -> LookupTypeMapResult a

lookupTypeMap :: Type -> TypeMap a -> [LookupTypeMapResult a]
lookupTypeMap ty tm
  = [ LTMR (fmap (lookup tsubst) prs) x
    | LkTR tsubst (MkMatch prs x) <- lkT (deBruijnize ty) (LkTR emptyTmplSubst tm) ]
  where
    lookup :: TmplSubst n -> (TmplVar, TmplKey n) -> (TmplVar, Type)
    lookup tsubst (tv, key) = (tv, lookupTmplSubst key tsubst)


{- *********************************************************************
*                                                                      *
                  GTypeMap
*                                                                      *
********************************************************************* -}

type GTypeMap :: (Nat -> Ty) -> Nat -> Ty
data GTypeMap a n
  = EmptyTM
  | TM { tm_tvar   :: Maybe (a (Succ n))  -- First occurrence of a template tyvar
       , tm_xvar   :: TmplOccs a n        -- Subsequent occurrence of a template tyvar

       , tm_bvar   :: BoundVarMap (a n)   -- Occurrence of a forall-bound tyvar
       , tm_fvar   :: FreeVarMap (a n)    -- Occurrence of a completely free tyvar

       , tm_fun    :: GTypeMap (GTypeMap a) n
       , tm_tycon  :: Map.Map TyCon (a n)
       , tm_forall :: GTypeMap a n
       }
deriving instance (forall m. Show (a m)) => Show (GTypeMap a n)

emptyGTypeMap :: GTypeMap a n
emptyGTypeMap = EmptyTM

mkEmptyGTypeMap :: GTypeMap a n
mkEmptyGTypeMap
  = TM { tm_tvar   = Nothing
       , tm_fvar   = emptyFreeVarMap
       , tm_xvar   = []
       , tm_bvar   = emptyBoundVarMap
       , tm_fun    = emptyGTypeMap
       , tm_tycon  = Map.empty
       , tm_forall = emptyGTypeMap }

type LkTResult :: (Nat -> Ty) -> Ty
data LkTResult a where
  LkTR :: forall n a. SNatI n => TmplSubst n -> a n -> LkTResult a

lkT :: forall a. DeBruijn Type -> LkTResult (GTypeMap a) -> [LkTResult a]
-- lk = lookup
lkT _ (LkTR _ EmptyTM)
  = []
lkT (D bv_env ty) (LkTR tsubst (TM { .. }))
  = tmpl_var_bndr ++ rest
  where
     rest = tmpl_var_occs ++ go ty

     go :: Type -> [LkTResult a]
     go (TyVarTy tv)
       | Just bv <- lookupCME tv bv_env = lkBoundVarOcc bv (tsubst, tm_bvar)
       | otherwise                      = lkFreeVarOcc  tv (tsubst, tm_fvar)
     go (FunTy t1 t2)    = concatMap (lkT (D bv_env t2)) $
                           lkT (D bv_env t1) (LkTR tsubst tm_fun)
     go (TyConTy tc)     = lkTC tc (tsubst, tm_tycon)

     go (ForAllTy tv ty) = lkT (D (extendCME tv bv_env) ty) (LkTR tsubst tm_forall)

     tmpl_var_bndr | Just x <- tm_tvar
--                   , null rest    -- This one line does overlap!
                   , noCaptured bv_env ty
                   = [LkTR (extendTmplSubst ty tsubst) x]
                   | otherwise
                   = []

     tmpl_var_occs = [ LkTR tsubst x
                     | (tmpl_var, x) <- tm_xvar
                     , deBruijnize (lookupTmplSubst tmpl_var tsubst)
                       `eqDeBT` (D bv_env ty)
                     ]

lkTC :: SNatI n => TyCon -> (TmplSubst n, Map.Map TyCon (a n)) -> [LkTResult a]
lkTC tc (tsubst, tc_map) = case Map.lookup tc tc_map of
                             Nothing -> []
                             Just x  -> [LkTR tsubst x]

xtT :: SNatI n
    => TmplVarSet -> DeBruijn Type
    -> (forall m. SNatI m => TmplKeys m -> XT (a m))
    -> TmplKeys n -> GTypeMap a n -> GTypeMap a n
-- xt = alter
xtT tmpls ty f tkeys EmptyTM
 = xtT tmpls ty f tkeys mkEmptyGTypeMap

xtT tmpls (D bv_env ty) f tkeys m@(TM {..})
  = go ty
  where
   go (TyVarTy tv)
      -- Second or subsequent occurrence of a template tyvar
      | Just xv <- Map.lookup tv tkeys  = m { tm_xvar = xtTmplVarOcc xv (f tkeys) tm_xvar }

      -- First occurrence of a template tyvar
      | tv `Set.member` tmpls = m { tm_tvar = f (extendTmplKeys tv tkeys) tm_tvar  }

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

liftXT :: SNatI n
       => (forall m. SNatI m => TmplKeys m -> GTypeMap a m -> GTypeMap a m)
       -> TmplKeys n -> Maybe (GTypeMap a n) -> Maybe (GTypeMap a n)
liftXT insert tkeys Nothing  = Just (insert tkeys emptyGTypeMap)
liftXT insert tkeys (Just m) = Just (insert tkeys m)


{- *********************************************************************
*                                                                      *
                   Template variables
*                                                                      *
********************************************************************* -}

type TmplVar = TyVar
type TmplVarSet = Set.Set TyVar
type TmplKey n = Fin n
type TmplKeys n = Map.Map TmplVar (TmplKey n)  -- Maps TmplVar :-> TmplKey

emptyTmplKeys :: TmplKeys n
emptyTmplKeys = Map.empty

extendTmplKeys :: forall n. SNatI n => TyVar -> TmplKeys n -> TmplKeys (Succ n)
extendTmplKeys tv tkeys = Map.insert tv (maxFin :: Fin (Succ n)) (bumpTmplKeysRange tkeys)

bumpTmplKeysRange :: TmplKeys n -> TmplKeys (Succ n)
bumpTmplKeysRange = fmap bumpFinIndex   -- should rewrite to a simple call to `coerce`

type TmplOccs :: (Nat -> Ty) -> Nat -> Ty
type TmplOccs a n = [(TmplKey n, a n)]

xtTmplVarOcc :: TmplKey n -> XT (a n) -> TmplOccs a n -> TmplOccs a n
xtTmplVarOcc key f []
  = xtCons key (f Nothing) []
xtTmplVarOcc key f ((key1,x):prs)
  | key == key1 = xtCons key (f (Just x)) prs
  | otherwise   = (key1,x) : xtTmplVarOcc key f prs

xtCons :: TmplKey n -> Maybe (a n) -> TmplOccs a n -> TmplOccs a n
xtCons _   Nothing  tmpl_occs = tmpl_occs
xtCons key (Just x) tmpl_occs = (key,x) : tmpl_occs

---------------
type TmplSubst n = FinMap n Type     -- Maps TmplKey -> Type

emptyTmplSubst :: TmplSubst n
emptyTmplSubst = emptyFinMap

lookupTmplSubst :: TmplKey n -> TmplSubst n -> Type
lookupTmplSubst key subst
  = case lookupFinMap key subst of
      Just ty -> ty
      Nothing -> error ("lookupTmplSubst " ++ show key)

extendTmplSubst :: forall n. SNatI n => Type -> TmplSubst n -> TmplSubst (Succ n)
extendTmplSubst ty subst
  = insertFinMap (maxFin :: Fin (Succ n)) ty (bumpFinMapIndex subst)

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

lkBoundVarOcc :: SNatI n => BoundVar -> (TmplSubst n, BoundVarMap (a n)) -> [LkTResult a]
lkBoundVarOcc var (tsubst, env) = case lookupBoundVarMap var env of
                                     Just x  -> [LkTR tsubst x]
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

lkFreeVarOcc :: SNatI n => TyVar -> (TmplSubst n, FreeVarMap (a n)) -> [LkTResult a]
lkFreeVarOcc var (tsubst, env) = case Map.lookup var env of
                                    Just x  -> [LkTR tsubst x]
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

instance Pretty (Fin n) where
  ppr f = ppr (finToInt f)

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

instance (forall n. Pretty (a n)) => Pretty (GTypeMap a n) where
  ppr EmptyTM = text "EmptyTM"
  ppr (TM {..}) = text "TM" <+> braces (vcat
                    [ text "tm_tvar =" <+> ppr tm_tvar
                    , text "tm_xvar =" <+> ppr tm_xvar
                    , text "tm_bvar =" <+> ppr tm_bvar
                    , text "tm_fvar =" <+> ppr tm_fvar
                    , text "tm_fun =" <+> ppr tm_fun
                    , text "tm_tycon =" <+> ppr tm_tycon
                    , text "tm_forall =" <+> ppr tm_forall ])
