-- TrieMap2 :: Singleton and empty in ExprMap
--             No DeBruijn, no SEMap, no matching

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
import qualified Text.Read as Read
import qualified Text.ParserCombinators.ReadP as ReadP

{- *********************************************************************
*                                                                      *
                   Expr
*                                                                      *
********************************************************************* -}

type Var    = String
type Lit    = String

data Expr = App Expr Expr
          | Var Var
          deriving( Eq )

{- *********************************************************************
*                                                                      *
                  The TrieMap class
*                                                                      *
********************************************************************* -}

class Eq (TrieKey tm) => TrieMap tm where
   type TrieKey tm :: Type
   emptyTM  :: tm v
   lookupTM :: TrieKey tm -> tm v -> Maybe v
   alterTM  :: TrieKey tm -> XT v -> tm v -> tm v
   foldTM   :: (v -> a -> a) -> tm v -> a -> a

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

foldMaybe :: (v -> a -> a) -> Maybe v -> a -> a
foldMaybe f Nothing  z = z
foldMaybe f (Just v) z = f v z

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

alterFreeVarOcc :: Var -> XT a -> FreeVarMap a -> FreeVarMap a
alterFreeVarOcc tv xt tm = Map.alter xt tv tm


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
   emptyTM  = emptyList
   lookupTM = lookupList
   alterTM  = alterList
   foldTM   = foldList

emptyList :: ListMap m a
emptyList = EmptyLM

lookupList :: TrieMap tm => [TrieKey tm] -> ListMap tm v -> Maybe v
lookupList _   EmptyLM = Nothing
lookupList key (LM {..})
  = case key of
      []     -> lm_nil
      (k:ks) -> lm_cons |> lookupTM k >=> lookupTM ks

alterList :: TrieMap tm => [TrieKey tm] -> XT v -> ListMap tm v -> ListMap tm v
alterList ks xt tm@(LM {..})
  = case ks of
      []      -> tm { lm_nil  = lm_nil |> xt }
      (k:ks') -> tm { lm_cons = lm_cons |> alterTM k |>> alterTM ks' xt }

foldList :: TrieMap tm => (v -> a -> a) -> ListMap tm v -> a -> a
foldList f (LM {..}) = foldMaybe f lm_nil . foldTM (foldList f) lm_cons


{- *********************************************************************
*                                                                      *
                  ExprMap
*                                                                      *
********************************************************************* -}

data ExprMap a
  = EmptyEM
  | SingleEM Expr a
  | EM { em_app  :: ExprMap (ExprMap a)
       , em_var  :: Map.Map Var a }
  deriving( Eq )

deriving instance (Show (TrieKey ExprMap), Show v)
               => Show (ExprMap v)

instance TrieMap ExprMap where
  type TrieKey ExprMap = Expr
  emptyTM  = mkEmptyExprMap
  lookupTM = lookupExpr
  alterTM  = alterExpr
  foldTM   = foldExpr

emptyExprMap :: ExprMap a
emptyExprMap = EmptyEM

mkEmptyExprMap :: ExprMap a
mkEmptyExprMap
  = EM { em_app  = emptyExprMap
       , em_var  = Map.empty }

lookupExpr :: Expr -> ExprMap v -> Maybe v
lookupExpr e EmptyEM
  = Nothing
lookupExpr e (SingleEM k v)
  | e == k    = Just v
  | otherwise = Nothing
lookupExpr e (EM { .. })
  = case e of
      Var x     -> em_var |> Map.lookup x
      App e1 e2 -> em_app |>  lookupTM e1 >=> lookupTM e2

alterExpr :: Expr -> XT v -> ExprMap v -> ExprMap v
alterExpr e xt m@(EM {..})
  = case e of
      Var x     -> m { em_var = alterFreeVarOcc  x  xt em_var }
      App e1 e2 -> m { em_app = em_app |> alterTM e |>> alterTM e2 xt }

foldExpr :: (v -> a -> a) -> ExprMap v -> a -> a
foldExpr f (EM {..})
  = foldFVM f em_var .
    foldTM (foldTM f) em_app


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
  ppr EmptyEM        = text "EmptySEM"
  ppr (SingleEM k v) = text "SingleSEM" <+> ppr k <+> ppr v
  ppr (EM {..}) = text "EM" <+> braces (vcat
                    [ text "em_var =" <+> ppr em_var
                    , text "em_app ="  <+> ppr em_app ])

