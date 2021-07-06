module Main where

import Criterion.Main
import Criterion.Types
import Bench

cfg = defaultConfig { csvFile = Just "crit.csv", reportFile = Just "crit.html" }
main = Criterion.Main.defaultMainWith cfg Bench.criterion
