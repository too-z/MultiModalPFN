# Multi-Modal PFN (Prior-data Fitted Network)


## Getting started
Install
```
python setup.py develop
```

python 3.10

### Usage

You need to place a checkpoint file and modify the model_path like below
```
model_path = Path(__file__).parent/ "parameters" / "tabpfn-v2-classifier.ckpt"
```

## License

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