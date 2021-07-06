module Main where

import Criterion.Main
import Criterion.Types
import Bench

cfg = defaultConfig { csvFile = Just "crit.new.csv", reportFile = Just "crit.new.html" }
main = Criterion.Main.defaultMainWith cfg Bench.criterion
