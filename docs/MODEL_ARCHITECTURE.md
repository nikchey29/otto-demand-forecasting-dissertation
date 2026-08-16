# Model architecture

## Transformer

The Transformer is designed for the size of the aggregated OTTO series rather than for the size
of the raw event log.

### Input

For each forecast origin the input has shape:

```text
(batch, 168, 8)
```

The eight channels are log clicks, log carts, log orders, hour sine/cosine, day sine/cosine and
the weekend flag.

### Encoder

The model first projects the eight features to 32 dimensions. Sinusoidal positional encoding is
added before two pre-normalised Transformer encoder layers.

Main settings:

| Setting | Value |
|---|---:|
| `d_model` | 32 |
| attention heads | 4 |
| encoder layers | 2 |
| feed-forward width | 128 |
| dropout | 0.15 |
| activation | GELU |

I use pre-normalisation (`norm_first=True`) because it gave more stable optimisation in the
small-data setting.

### Attention pooling

Rather than using only the final encoded time step, a small scoring network assigns a normalised
weight to each of the 168 encoded hours. The weighted sum becomes the context vector used by
the forecast head.

This was a deliberate choice: the relevant signal may be a recurring hour from earlier in the
week rather than only the most recent observation.

### Forecast head

The pooled vector is passed through a two-layer feed-forward head and reshaped to:

```text
(batch, 24, 2)
```

so carts and orders for all 24 future hours are predicted directly.

## GRU comparison

The GRU baseline uses two recurrent layers with a hidden size of 48. The final hidden state is
mapped to the same 24 × 2 output shape. Keeping the output formulation identical makes the
comparison with the Transformer easier to interpret.

## Training

Both neural models use:

- Huber loss;
- AdamW;
- gradient clipping;
- learning-rate reduction on validation loss;
- early stopping;
- deterministic seeds where supported.

The code is in:

- `src/otto_forecasting/model.py`
- `src/otto_forecasting/training.py`

The self-contained Colab notebook contains the same core Transformer architecture so the model
can be inspected and run without first installing the package.
