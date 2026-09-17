{
  description = "python fixups for use with uvpart";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };
  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      debug = true;
      imports = [
        inputs.treefmt-nix.flakeModule
      ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          pkgs,
          ...
        }:
        {
          treefmt = {
            programs = {
              nixfmt.enable = true;
              mdformat.enable = true;
            };
          };
        };
      flake = {
        flakeModule = ./flake-module.nix;
        fixup-overlay = pkgs: pkgs.callPackage ./fixup-overlay.nix { };
        # Opt-in: needs the extension list for the project that uses it.
        #   (inputs.uvpart-fixups.gptqmodel-ops {
        #     inherit pkgs;
        #     extensions = [ "marlin_fp16" "marlin_bf16" ];
        #   })
        gptqmodel-ops = args: import ./gptqmodel-ops.nix args;
      };
    };
}
