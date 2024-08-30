# LASPATED: Library for Analysis of Spatio-Temporal Discrete Data

This repository contains the replication code for the results in "LASPATED: A Library for the Analysis of
Spatio-Temporal Discrete Data" paper ([Arxiv link](https://arxiv.org/abs/2401.04156)). The library source code is available at [Github](https://github.com/vguigues/LASPATED/)

To run the replication script check this [guide](Replication/README.md). The installation process can also be done via Docker.


## Docker image


If you have a [Gurobi](https://gurobi.com) Web License, you can build the container with Gurobi support by running:

```
docker build --build-arg USE_GUROBI=1 -t laspated .
```

The above command will build the container with Gurobi 11.0.1 installed. If you don't have a Gurobi license, just pass USE_GUROBI=0, in this case the experiments with the model with covariates will not be performed.

To run the container with Gurobi support, you can pass the license to the container with:
```
docker run --volume="/absolute/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro" -it laspated
```

This will open a shell environment with all dependencies installed. Once in the container environment you can run the replication script with:
```
cd Replication
python replication_script.py
```


### DockerHub

The docker image is also available at [DockerHub](https://dockerhub.com). To install it, just run:


```
docker pull victorvhrn/laspated_replication
```


**Note:** The DockerHub container is built without Gurobi support by default. To enable Gurobi support, you need to recompile the C++ code with Gurobi support inside the container environment. Check the [Replication README](Replication/README.md) on how to recompile the code with Gurobi support.



