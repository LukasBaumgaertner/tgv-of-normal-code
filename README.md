This repository contains code for mesh denoising with TGV of the normal for the Paper

Total Generalized Variation of the Normal Vector Field and Applications to Mesh Denoising (https://arxiv.org/abs/2507.13530)  <br>
by L. Baumgärtner, R. Bergmann, R. Herzog, S. Schmidt, M. Weiß.

The code requires the latest installation of (legacy) fenics.
In order to resolve the dependencies, it is advised to run the code inside a Docker container, following the included `Dockerfile`: 
```bash
docker build -t tgv_of_normal .
```

To add orientation to the .obj files for the respective testcases, run e.g.
```bash
docker run -ti --rm -v $(pwd):/home/fenics/shared -w /home/fenics/shared/Fandisk tgv_of_normal python3 add_orientation.py
```
Afterwards, to run an experiment run e.g. 
```bash
docker run -ti --rm -v $(pwd):/home/fenics/shared -w /home/fenics/shared/Fandisk/Newton tgv_of_normal python3 test.py
```

Afterwards, the results can be viewed in paraview.
