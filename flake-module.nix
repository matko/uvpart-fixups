{
  perSystem =
    { pkgs, ... }:
    let
      fixup-overlay = pkgs.callPackage ./fixup-overlay.nix { };
    in
    {
      uvpart.pythonOverlays = [ fixup-overlay ];
    };
}
