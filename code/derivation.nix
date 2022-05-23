{ mkDerivation, base, containers, criterion, deepseq, ghc-heap
, hashable, lib, microlens, microlens-ghc, pretty, QuickCheck
, transformers, tree-view, unordered-containers
}:
mkDerivation {
  pname = "triemap";
  version = "0.1";
  src = ./.;
  libraryHaskellDepends = [
    base containers criterion deepseq ghc-heap hashable microlens
    microlens-ghc pretty QuickCheck transformers tree-view
    unordered-containers
  ];
  testHaskellDepends = [
    base containers QuickCheck transformers tree-view
  ];
  benchmarkHaskellDepends = [ base criterion ];
  doHaddock = false;
  doBenchmark = true;
  license = lib.licenses.bsd3;
}
