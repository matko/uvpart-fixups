{ stdenv, gcc }:
stdenv.mkDerivation {
  name = "cuda-loader-helper";
  src = ./.;
  nativeBuildInputs = [ gcc ];
  buildPhase = ''
    mkdir -p build
    gcc -shared -fPIC -o build/cuda_loader_helper.so cuda_loader_helper.c -ldl
  '';

  installPhase = ''
    mkdir -p $out/lib
    cp build/cuda_loader_helper.so $out/lib/
  '';
}
