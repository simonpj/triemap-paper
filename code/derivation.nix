{ mkDerivation, base, containers, criterion, deepseq, ghc-heap
, hashable, lib, microlens, microlens-ghc, pretty, QuickCheck
, transformers, tree-view, unordered-containers, cabal-install
}:
mkDerivation {
  pname = "triemap";
  version = "0.1";
  src = ./.;
  buildDepends = [
    cabal-install
  ];
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
