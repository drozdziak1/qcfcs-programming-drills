{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { ... }@inputs: inputs.flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import inputs.nixpkgs {
      inherit system;
    };
    bydon = pkgs.python3.withPackages (ps: with ps; [
      ipython
      ]);
  in {
    devShell = pkgs.mkShell {
      nativeBuildInputs = [bydon];
    };
  });
}
