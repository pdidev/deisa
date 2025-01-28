{
  pdi,
  fetchFromGitHub,
}:
pdi.overrideAttrs (old: {
  src = fetchFromGitHub {
    owner = "pdidev";
    repo = "pdi";
    rev = "105161d5c93431d674c73ef365dce3eb724b4fcb";
    hash = "sha256-EUqgscpCubub2Zl/7fcgbdVx216Y2Ke7h8Zui2SieP8=";
  };
  cmakeFlags = [
    # Force using nix gbenchmark instead of vendored version
    "-DUSE_benchmark=SYSTEM"
    "-DBUILD_DEISA_PLUGIN=ON"
    "-DBUILD_PYTHON=ON"
    "-DBUILD_PYCALL_PLUGIN=ON"
  ];
})
