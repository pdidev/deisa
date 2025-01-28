let
  pkgs =
    import
    (
      fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz"
    ) {};

  myPython = pkgs.python3;

  pdi_with_deisa = pkgs.callPackage ./pdi_deisa_plugin.nix {};
in
  pkgs.mkShell {
    LD_LIBRARY_PATH = "${pdi_with_deisa}/lib";
    packages = [
      # need the python packages
      (myPython.withPackages (pp: [
        pp.distributed
        pp.dask
        pp.bokeh
        pp.graphviz
        pp.pytest
        pp.pip
      ]))

      pdi_with_deisa

      # needed to use mpi and to compile the example project
      pkgs.mpi
      pkgs.gcc
      pkgs.cmake

      # needed by pdi
      pkgs.libyaml

      # needed so that cmake in pdi can find yaml
      pkgs.pkg-config

      # Optional
      # I use pyright for type checking
      pkgs.pyright
    ];
  }
