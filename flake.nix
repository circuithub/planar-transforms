{
  description = "PyTorch planar transforms library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    inherit (nixpkgs) lib;

    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];

    legacyPackages = lib.genAttrs supportedSystems (
      system:
        import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        }
    );
  in {
    devShells = lib.genAttrs supportedSystems (system: let
      pkgs = legacyPackages.${system};
      python = pkgs.python312;
      pythonPackages = python.pkgs;
    in {
      default = pkgs.mkShell {
        env = {
          FORCE_COLOR = "1";
        };

        buildInputs = [
          python
          pythonPackages.pip
          pythonPackages.virtualenv
          pythonPackages.torch
          pythonPackages.torchvision
          pythonPackages.black
          pythonPackages.ruff
          pythonPackages.mypy
          pythonPackages.pytest
          pythonPackages.pytest-cov
          pythonPackages.hypothesis
        ];

        shellHook = ''
          export PYTHONPATH="$PWD/src:$PYTHONPATH"
          echo "planar-transforms dev shell"
          echo "Available commands: black, ruff, mypy, pytest"
        '';
      };
    });

    packages = lib.genAttrs supportedSystems (system: let
      pkgs = legacyPackages.${system};
      pythonPackages = pkgs.python312.pkgs;
    in {
      default = pythonPackages.buildPythonPackage {
        pname = "planar-transforms";
        version = "0.1.0";
        src = ./.;
        format = "pyproject";

        nativeBuildInputs = [pythonPackages.hatchling];

        propagatedBuildInputs = [
          pythonPackages.torch
          pythonPackages.torchvision
        ];

        pythonImportsCheck = ["planar_transforms"];
      };
    });

    formatter = lib.genAttrs supportedSystems (system: legacyPackages.${system}.alejandra);
  };
}
