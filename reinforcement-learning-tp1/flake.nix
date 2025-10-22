{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            lolcat
            ruff
            mypy
            (
              python313.withPackages
              (
                ppkgs:
                  with ppkgs; [
                    ipython
                    ipykernel
                    jupyter-all
                    numpy
                    matplotlib
                    scipy
                    pytest
                  ]
              )
            )
          ];
        };
      }
    );
}
