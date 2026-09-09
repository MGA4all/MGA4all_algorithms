<img src="./docs/mga4all_logo_algos.png" alt="MGA4all_logo"  width="100">

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# MGA4all - algorithms
Various Modelling to Generate Alternative (MGA) algorithms for different energy system optimisation modelling frameworks

## Implemented Algorithms and supported model backends

| Algorithm                                                         | PyPSA + linopy |
|-------------------------------------------------------------------|----------------|
| [SPORES](https://doi.org/10.1016/j.joule.2020.08.002)             | ✅             |
| [Random Directions](https://doi.org/10.1016/j.energy.2017.03.043) | ✅             |
| [Hop-Skip-JUmp](https://doi.org/10.1016/j.eneco.2010.05.002)      | ✅             |   

## Running MGA4all

We separate the MGA algorithms (e.g., `random_directions`) and modelling backends (e.g. `pypsa`). You can choose the MGA algorithm that best fits your needs and, depending on the modelling framework you want to work with, choose the appropriate backend.

We prefer using [`hatch`](https://hatch.pypa.io/latest/install/)
(>=1.16) to create/manage necessary environments and run commands

```
$ hatch run <command> [options]
```

Where, `<command>` can be any script that uses `MGA4all`; by default
the `pypsa` backend is used.

If you don't want to use `hatch`, create a virtual environment as you
would, install MGA4all in edit mode:

```
$ pip install -e .
```

and run your script as you normally would.

### Testing with included examples

MGA4All also includes an example PyPSA model.  
A user can use this model for testing while working with MGA4All
interactively in a Python shell.

```python
import yaml

from mga4all.mga_random_directions import random_directions_algorithm
from mga4all.examples import create_pypsa_network


with open("configs/test_config_random_directions.yaml") as yf:
    test_config = yaml.safe_load(yf)

mynetwork = create_pypsa_network()
mynetwork.optimize(sovler_options={'solver_name': 'highs'})

mga_alternatives, mga_spatial_alternatives = random_directions_algorithm(test_config, mynetwork)
```

### Tutorials

In `docs\tutorials`, you will find several jupyter notebooks that showcase how to use different MGA algorithms on the above example PyPSA model, and how to inspect their outputs.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChristianDDinga"><img src="https://avatars.githubusercontent.com/u/127748593?v=4?s=100" width="100px;" alt="Christian Doh Dinga"/><br /><sub><b>Christian Doh Dinga</b></sub></a><br /><a href="#code-ChristianDDinga" title="Code">💻</a> <a href="#data-ChristianDDinga" title="Data">🔣</a> <a href="#ideas-ChristianDDinga" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-ChristianDDinga" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

### Licence

Copyright MGA4all (Contributors)[https://github.com/MGA4all/MGA4all_algorithms/graphs/contributors]

MGA4all is licensed under the open source [MIT License](/LICENSE). 
