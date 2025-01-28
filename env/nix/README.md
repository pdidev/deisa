# Deisa DevEnv Using Nix

If you want to quickly setup a development environment to change some of the default behavior of 
Deisa, you can do so easily with Nix. This is useful for advanced users who wish to customize and 
test Deisa. 

If you already have Nix installed, then the `shell.nix` file will install all the necessary 
dependencies of Deisa.

Simply run: 
```bash
# from this directory 
$ nix-shell .
# from root of project
$ nix-shell env/nix/shell.nix
``` 
However, you also have to add path to the root of the project to `PYTHONPATH`:
```
export PYTHONPATH=/path/to/deisa/root
```
Alternatively, if you are also using `nix-direnv`, just create a `.envrc` file at the root of the 
project which contains the following:
```
use nix env/nix/shell.nix
export PYTHONPATH=$PWD
```
Then, simply run `direnv allow` in the terminal. Now, everytime you enter the Deisa directory, the 
development environment will be automatically activated and the `PYTHONPATH` will be correctly set.

Happy hacking!
