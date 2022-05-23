{ nixpkgs ?
    import (builtins.fetchTarball {
      # Descriptive name to make the store path easier to identify
      name = "nixpkgs-unstable-2021-07-06";
      # Commit hash for nixos-unstable as of 2018-09-12
      url = "https://github.com/nixos/nixpkgs/archive/dfd82985c273aac6eced03625f454b334daae2e8.tar.gz";
      # Hash obtained using `nix-prefetch-url --unpack <url>`
      sha256 = "1ipd1k1gvxh9sbg4w4cpa3585q09gvsq8xbjvxnnmfjib6r6xx4i";
    }) {} }:
(nixpkgs.pkgs.haskellPackages.callPackage ./derivation.nix { }).env
