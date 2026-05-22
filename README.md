# License

This work is released as free software under the GNU Public License.  All constituent source code files are covered by this license.  See the file COPYING for the full legal details of the license.

# Usage

To use this software, first set up a python environment with its dependencies.

## Environment setup with uv (recommended)

From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r cc_requirements.txt
```

If you need to recreate the virtual environment from scratch:

```bash
uv venv --python 3.12 --clear .venv
source .venv/bin/activate
uv pip install -r cc_requirements.txt
```

The `environment.yml` file is still available if you prefer using conda-compatible tools instead of `uv`.

For all scripts, run with `-h` or `--help` for some information on usage and arguments.  Most scripts are configured using the hydra library [https://hydra.cc/], and so the standard hydra CLI allows you to modify their configuration with command-line arguments.

All commands below assume you are in the repository root and invoke scripts as Python modules (`python -m package.module`) rather than as file paths (`python path/to/file.py`), so that sibling packages like `simulation/` and `world/` resolve on `sys.path`.  Running the scripts directly as files will fail with `ModuleNotFoundError: No module named 'simulation'`.

## Training

If you're not using the pre-trained model weights (information on how to get them in the "Model Weights" section), you'll need to train your own model.  To generate a training dataset, use the `simulation/citygraph_dataset.py` script:

```bash
python -m simulation.citygraph_dataset --min N --max N --n NUM_GRAPHS /path/to/output/dataset
```

Note that right now, there's a bug for running the algorithm on batches of graphs with different numbers of nodes, so you should pass the same value to `--min` and `--max` to make sure all graphs in the dataset have the same size.  The dataset will be output to the directory you specify.

To train a model, use the script `learning/inductive_route_learning.py`.  You will need to specify the path to your generated training dataset directory as follows:

```bash
python -m learning.inductive_route_learning dataset.kwargs.path=/path/to/your/dataset
```

By default, the model will be trained over a range of cost weights from 0 to 1.  To train just on an operator perspective setting, add the argument `experiment/cost_function=op`,
or to train on a passenger perspective setting, add `experiment/cost_function=pp`.

Training should take around 3-6 hours on a modern commercial GPU.

You can optionally add the argument `+run_name=my_run_name` to name the training run, which will affect the name of the tensorboard logs (stored by default in a directory called `training_logs`) and the name of the output weight file.  If this is not provided, the current date and time will be used as the name of the run.  

When training is complete, the trained weights will be stored in the directory `output` in a file named `inductive_[run-name].pt`.
 
## Evaluation

We mainly evaluate our methods on the Mandl and Mumford datasets, which can be downloaded as a single archive (`CEC2013Supp.zip`) from [Christine Mumford's website](https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/Outline.html).  Download the archive and extract it to a directory on your system; the loader expects an `Instances/` subdirectory containing files like `MandlCoords.txt`, `MandlTravelTimes.txt`, `MandlDemand.txt`, and analogous triplets for `Mumford0`–`Mumford3`.

If the Cardiff URL above is unresponsive, the archive is mirrored on the [Internet Archive Wayback Machine](https://web.archive.org/web/*/users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/CEC2013Supp.zip).  To download the raw bytes of a Wayback snapshot without the rendering wrapper, append `id_` to the timestamp, e.g.:

```bash
curl -L -o CEC2013Supp.zip \
  "https://web.archive.org/web/20251216000000id_/https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/CEC2013Supp.zip"
```

Each script described in this section prints a line of comma-separated statistics about the best transit network it finds, with the header format:
,cost,C_p (minutes),C_o (minutes),d_0,d_1,d_2,d_{un},# disconnected node pairs,# stops out of bounds,running time (seconds),number of iterations

Each also saves the best transit network as a pickled torch tensor which can be read by other scripts, in a directory called `output_routes`.  The filename will contain the run name that can be provided to each script with `+run_name=my_run_name`.  If no run name is provided, the date and time when the script was launched will be used instead.

To evaluate a model on a Mumford city, use the script `learning/eval_route_generator.py`.  You must provide a `.pt` file with model weights, the path to the `Instances` sub-directory of the mumford dataset, and the name of the city on which to evaluate (`mandl` or `mumford0` - `mumford3`), as follows:
```bash
python -m learning.eval_route_generator \
  +model.weights=path_to_weights.pt \
  eval.dataset.path=/path/to/mumford/Instances \
  +eval=mandl \
  +run_name=my_mandl_lc100
```

To run the evolutionary algorithm (EA) on a city using the network generated by the above LC-100 run, the signature is similar, but without model weights:
```bash
python -m learning.bee_colony \
  eval.dataset.path=/path/to/mumford/Instances \
  +eval=mandl \
  init.path=output_routes/nn_construction_my_mandl_lc100_routes.pkl
```

And to run the neural evolutionary algorithm (NEA), use the same script but specify the `neural_bco_mumford` config file, and provide model weights and the path to the transit network from LC-100 to be used as the starting network:
```bash
python -m learning.bee_colony --config-name neural_bco_mumford \
  +model.weights=path_to_weights.pt \
  eval.dataset.path=/path/to/mumford/Instances \
  +eval=mandl \
  init.path=output_routes/nn_construction_my_mandl_lc100_routes.pkl
```

Note that "bee colony" is a holdover from an earlier stage in this research project, where we were using a "bee colony optimization" algorithm.

# Model weights

Model weights used for the ITSC experiments can be downloaded from the following link:
https://www.cim.mcgill.ca/~mrl/projs/transit_learning/itsc_2023

Those used for the most up-to-date PPO experiments (forthcoming) can be downloaded from: 
https://www.cim.mcgill.ca/~mrl/projs/transit_learning/ppo_2025

# Citation

If you make use of this code for academic work, please cite our associated conference paper, "Augmenting Transit Network Design Algorithms with Deep Learning":

```
@inproceedings{holliday2024autonomous,
    author = {Holliday, Andrew and Dudek, Gregory},
    title = {A Neural-Evolutionary Algorithm for Autonomous Transit Network Design},
    year = {2024},
    booktitle = {presented at 2024 IEEE International Conference on Robotics and Automation (ICRA)},
    organization = {IEEE}
}
```
