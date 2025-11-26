```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# {octicon}`code-square` API

```{eval-rst}
.. module:: Heimdall

.. automodule:: Heimdall
   :noindex:
```

## Trainer

Entry points for implementations of scFMs via Heimdall.

```{eval-rst}
.. module:: Heimdall.trainer
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    trainer.HeimdallTrainer
```

## $F_\textbf{G}$

Implementations of gene identity encodings.

```{eval-rst}
.. module:: Heimdall.fg
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    fg.Fg
    fg.IdentityFg
    fg.PretrainedFg
    fg.TorchTensorFg
    fg.CSVFg
```

## $F_\textbf{E}$

Implementations of gene expression encodings.

```{eval-rst}
.. module:: Heimdall.fe
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    fe.Fe
    fe.IdentityFe
    fe.ZeroFe
    fe.BinningFe
    fe.ScBERTBinningFe
```

## $F_\textbf{C}$

Implementations of single-cell representations.

```{eval-rst}
.. module:: Heimdall.fc
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    fc.Fc
    fc.DummyFc
    fc.ChromosomeAwareFc
```

## $\rm{O\small{RDER}}$

Implementations of ordering function for producing an order for gene tokens.

```{eval-rst}
.. module:: Heimdall.order
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    order.Order
    order.ExpressionOrder
    order.RandomOrder
    order.ChromosomeOrder
```

## $\rm{S\small{EQUENCE}}$

Implementations of sequence function for producing sequence of gene + cell metadata tokens.

```{eval-rst}
.. module:: Heimdall.tailor
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    tailor.Tailor
    tailor.ReorderTailor
    tailor.ChromosomeTailor
    tailor.WeightedResampleTailor
    tailor.ChromosomeBlockTailor
```

## $\rm{R\small{EDUCE}}$

Implementations of reduction operations for combining gene identity and expression encodings.

```{eval-rst}
.. module:: Heimdall.reduce
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    reduce.Reduce
    reduce.IdentityReduce
    reduce.SumReduce
    reduce.ChromosomeReduce
    reduce.ChromosomeSumReduce
```

## Task

Definition of pretraining and downstream tasks for single-cell foundation models.

```{eval-rst}
.. module:: Heimdall.task
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    task.Task
    task.SingleInstanceTask
    task.PairedInstanceTask
	task.SeqMaskedMLMTask
    task.Tasklist
```

## Model

Entry points for implementations of scFMs via Heimdall.

```{eval-rst}
.. module:: Heimdall.models
.. currentmodule:: Heimdall

.. autosummary::
    :toctree: api/
    :recursive:

    models.HeimdallModel
```
