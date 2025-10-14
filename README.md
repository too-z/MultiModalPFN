# Multi-Modal PFN (Prior-data Fitted Network)


## Getting started
Install
```
python setup.py develop
```

Place the checkpoint file and dataset in their respective locations, then update the model_path as shown below:

```
ln -s /path/to/model/params # symlink parameter
ln -s /path/to/data # symlink data

model_path = Path(__file__).parent/ "parameters" / "tabpfn-v2-classifier.ckpt"
```

## License
This project follows the original TabPFN license policy.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Copy tsp/datasets/openml directory into openml cache (default: ~/.cache/)
