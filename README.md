# muTransformers

This repo implements [muP](https://arxiv.org/abs/2203.03466) for selected PyTorch models in [Huggingface Transformers](https://github.com/huggingface/transformers).
The primary purpose of this repo is as a clean demonstration of how to inject muP into different variants of transformers.
As a secondary purpose, one can also use the models here as provided.

## Installation

Go to this project directory and do
```
pip install .
pip install mup
```

## Injecting muP into Existing Transformers

Taking BERT as an example, there are two files `modeling_bert.py` and `configuration_bert.py` in `mutransformers/models/bert/` we copied from [Huggingface Transformers](https://github.com/huggingface/transformers) and made a small number of modifications to implement muP.
Our modifications in these files can all be found by searching for `### muP`.

These files are copied from [Huggingface Transformers v4.16.2](https://github.com/huggingface/transformers/tree/v4.16.2). We provide the original files as `_original_*.py` for easy comparison, for example, `_original_modeling_bert.py`.

## Coord Check

[Coordinate checking](https://github.com/microsoft/mup#coord-check) is a way of verifying that muP is implemented correctly just like gradient checking is a way of verifying that autograd is implemented correctly.
You can find the coord check results in `tests/coordcheck/CoordCheck.ipynb`.
You can rerun the notebook yourself as well after installation.

## Basic Usage of Models
The models here can be used for your training purposes as well, though we have not made sure to replicate the original numbers of each of these transformer models.
The models in this package can be used as follows, taking BERT as an example:
```python
from mutransformers import BertConfig, BertForMaskedLM
from mup import make_base_shapes, set_base_shapes
from functools import partial
# define a base model
base_config = BertConfig(
    hidden_size=256,
    intermediate_size=256,
    num_attention_heads=16,
)
base_model = BertForMaskedLM(config=base_config)
# define a delta models where we vary all "widths" we want to vary
delta_config = BertConfig(
    hidden_size=200,
    intermediate_size=300,
    num_attention_heads=5,
)
delta_model = BertForMaskedLM(config=delta_config)
# define a base shape object based on comparing delta_model against base_model
base_shapes = make_base_shapes(base_model, delta_model, savefile='bert256.bsh')

# define target model
target_config = BertConfig(
    hidden_size=1024,
    intermediate_size=1024*4,
    num_attention_heads=32,
)
target_model = BertForMaskedLM(config=target_config)

# set base shapes
set_base_shapes(target_model, base_shapes)
# you can alternatively load base shape from file
# set_base_shapes(target_model, 'bert256.bsh')

# re-initialize
target_model.apply(target_model._init_weights)

# train target_model, etc
```

For more general information on how to use `mup`, see [the muP package documentation](https://github.com/microsoft/mup#basic-usage).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.