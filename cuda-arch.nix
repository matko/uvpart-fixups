# Rendering for uvpart.cudaArch, which carries a list of compute capabilities such as
# [ "8.9" ] or [ "8.0" "8.6" "8.9" "9.0" ].
#
# Every consumer spells the same thing differently: torch-based builds want
# "8.0;8.6;8.9;9.0+PTX", flash-attn wants "80;86;89;90", and llama.cpp's
# CMAKE_CUDA_ARCHITECTURES wants "80;89". So the option carries the capabilities
# themselves and each build renders its own form through here. One source means they cannot
# drift, which is otherwise how a package ends up compiled for an architecture the
# environment is not set up to run on.
{ lib }:
let
  # PTX is appended to the last entry only, so a single-entry list -- the common case,
  # one machine -- still runs on cards newer than the list. On the card named, the
  # ahead-of-time cubin is what executes; PTX is for everywhere else.
  withPtx = list: lib.imap0 (i: a: if i == builtins.length list - 1 then "${a}+PTX" else a) list;

  splitCap =
    c:
    let
      parts = lib.splitString "." c;
    in
    {
      major = builtins.elemAt parts 0;
      minor = lib.toInt (builtins.elemAt parts 1);
    };

  # The closest architecture a package can actually build, for one requested capability.
  # A cubin runs on a card of the same major version with an equal or higher minor, so the
  # answer is the highest entry of the same major that is not above the request: asking for
  # 8.9 of a package that knows only 8.0 gets 8.0, and that cubin runs on Ada. An entry
  # from a *higher* major would not run at all, and a request from a major the package has
  # no entry for has no compatible answer at all -- both are reported as null.
  snapOne =
    supported: c:
    let
      requested = splitCap c;
      candidates = builtins.filter (
        x: (splitCap x).major == requested.major && (splitCap x).minor <= requested.minor
      ) supported;
      byMinorDesc = builtins.sort (a: b: (splitCap a).minor > (splitCap b).minor) candidates;
    in
    if byMinorDesc == [ ] then null else builtins.head byMinorDesc;
in
{
  toTorchCudaArchList = list: lib.concatStringsSep ";" (withPtx list);
  toArchNumbers = list: map (lib.replaceStrings [ "." ] [ "" ]) list;
  toJoinedArchNumbers = list: lib.concatStringsSep ";" (map (lib.replaceStrings [ "." ] [ "" ]) list);

  # Snapping for a package whose build knows only a fixed set of architectures -- flash-attn
  # emits gencode for 80/90/100/120 and nothing else, so it cannot be handed an arbitrary
  # capability. Capabilities with no compatible entry are dropped; if none survive, the
  # fallback stands, which is the value that package used before this option existed.
  snapToSupported =
    supported: fallback: caps:
    let
      snapped = builtins.filter (c: c != null) (map (snapOne supported) caps);
    in
    if snapped == [ ] then fallback else snapped;
}
