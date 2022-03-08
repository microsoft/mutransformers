# Copyright 2022 Microsoft Corporation.

from functools import partial
from itertools import cycle

import numpy as np
import torch

import seaborn as sns
from mup.coord_check import get_coord_data, plot_coord_data
from mup import set_base_shapes, make_base_shapes
from transformers import BertTokenizer, GPT2Tokenizer

from mutransformers import BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM, GPT2Config, GPT2LMHeadModel

sns.set()

def get_dataloader(arch):
  if arch in ('bert', 'roberta'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = tokenizer("The capital of France is [MASK].", return_tensors="pt")['input_ids']
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    dataloader = cycle([dict(input_ids=input_ids, labels=labels)])
  elif arch == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "The capital of France is Paris."
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input['labels'] = encoded_input['input_ids']
    dataloader = cycle([encoded_input])
  return dataloader

def make_bsh(arch, filename=None):
  if arch == 'roberta':
    base_config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        type_vocab_size=1,
    )
    delta_config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=200,
        intermediate_size=200,
        num_attention_heads=5,
        num_hidden_layers=2,
        type_vocab_size=1,
    )
    base_model = RobertaForMaskedLM(config=base_config)
    delta_model = RobertaForMaskedLM(config=delta_config)
  elif arch == 'bert':
    base_config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        type_vocab_size=1,
    )
    delta_config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=200,
        intermediate_size=200,
        num_attention_heads=5,
        num_hidden_layers=2,
        type_vocab_size=1,
    )
    base_model = BertForMaskedLM(config=base_config)
    delta_model = BertForMaskedLM(config=delta_config)
  elif arch == 'gpt2':
    base_config = GPT2Config(
      n_head=4,
      activation_function='relu',
      n_embd=256,
      n_layer=2,
      # num_labels=1,
    )
    delta_config = GPT2Config(
      n_head=5,
      activation_function='relu',
      n_embd=200,
      n_layer=2,
      # num_labels=1,
    )
    base_model = GPT2LMHeadModel(config=base_config)
    delta_model = GPT2LMHeadModel(config=delta_config)
  else:
    raise NotImplementedError()
  base_shapes = make_base_shapes(base_model, delta_model, savefile=filename)
  return base_shapes

def get_lazy_model(arch, width, base_shape=None, mup=True, readout_zero_init=True, query_zero_init=True, vary_nhead=False):
  width = int(width)
  nhead = 4
  if vary_nhead:
    nhead = int(4 * width / 252)
  def f():
    if arch == 'roberta':
      config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=width,
        intermediate_size=width,
        num_attention_heads=nhead,
        num_hidden_layers=2,
        type_vocab_size=1,
        attn_mult=8 if mup else None,
        classifier_dropout=0
      )
      model = RobertaForMaskedLM(config=config)
    elif arch == 'bert':
      config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        hidden_size=width,
        intermediate_size=width,
        num_attention_heads=nhead,
        num_hidden_layers=2,
        type_vocab_size=1,
        attn_mult=8 if mup else None,
        classifier_dropout=0
      )
      model = BertForMaskedLM(config=config)
    elif arch == 'gpt2':
      config = GPT2Config(
        n_head=nhead,
        activation_function='relu',
        n_embd=width,
        n_layer=2,
        attn_mult=8 if mup else None,
        # num_labels=1,
        # resid_pdrop=0,
        # embd_pdrop=0,
        # attn_pdrop=0,
      )
      model = GPT2LMHeadModel(config=config)
    if mup:
      set_base_shapes(model, base_shape)
    else:
      set_base_shapes(model, None)

    model.apply(
      partial(model._init_weights,
              readout_zero_init=readout_zero_init,
              query_zero_init=query_zero_init,
              ))
    return model
  return f

def plot_coord_check(arch, mup=True, vary_nhead=False, y='l1', widths=None, optimizer='adam',
                    nseeds=1, nsteps=4, loglog=False, logbase=2, legend=None,
                    **get_coord_data_kw):
  if widths is None:
    widths = 2**np.arange(6, 11)
  base_shape = make_bsh(arch)
  models = {width: get_lazy_model(arch, width, base_shape=base_shape, mup=mup, vary_nhead=vary_nhead) for width in widths}
  dataloader = get_dataloader(arch)
  df = get_coord_data(models, dataloader, mup=mup, optimizer=optimizer,
                      nseeds=nseeds, dict_in_out=True,
                      nsteps=nsteps, **get_coord_data_kw)

  prm = 'mup' if mup else 'sp'
  width = 'nhead' if vary_nhead else 'dhead'
  return plot_coord_data(df, legend=legend, loglog=loglog, logbase=logbase, y=y,
        save_to=f'{arch}_{prm}_{width}_coord_check.png',  suptitle=f'{prm} {arch} {width}',
        face_color='xkcd:light grey' if not mup else None)