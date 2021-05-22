{- *********************************************************************
*                                                                      *
                   Preliminaries
*                                                                      *
********************************************************************* -}

module Prelim
  ( Ty
  , Nat(..)
  ) where

import qualified Data.Kind as Kind
import Data.Type.Equality ( (:~:)(..) )
import Unsafe.Coerce ( unsafeCoerce )

type Ty = Kind.Type

{- *********************************************************************
*                                                                      *
                   Natural numbers
*                                                                      *
********************************************************************* -}

data Nat = Zero | Succ Nat    -- used only at compile time
