-- horrible hack to be able to check types in lhs2TeX.
-- must be a separate module because of TH stage restriction

module RandomType where

import Language.Haskell.TH
import System.Random

randomType :: Q Type
randomType = LitT <$> (NumTyLit <$> (abs <$> randomIO))
