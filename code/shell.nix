{ nixpkgs ?
    import (builtins.fetchTarball {
      # Descriptive name to make the store path easier to identify
      name = "nixpkgs-unstable-2021-07-06";
      # Commit hash for nixos-unstable as of 2018-09-12
      url = "https://github.com/nixos/nixpkgs/archive/8eb54b8e0907ce37ba0eeeacfb67676d42d857bc.tar.gz";
      # Hash obtained using `nix-prefetch-url --unpack <url>`
      sha256 = "09pzj1122ha9rbb04m395a1wv35hjz02dq5lxs3kglndjmapc2nv";
    }) {}
, compiler ? "ghc8104" }:
(nixpkgs.pkgs.haskell.packages.${compiler}.callPackage ./derivation.nix { }).env
