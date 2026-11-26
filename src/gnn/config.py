"""Firma del modello per MLflow.

Descrive la forma dei tensori in ingresso e in uscita, ed e' registrata insieme al modello
da `mlflow.pytorch.log_model` in train.py e validation_train.py.
"""
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 30), name="x"),
                       TensorSpec(np.dtype(np.float32), (-1, 11), name="edge_attr"),
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index"),
                       TensorSpec(np.dtype(np.int32), (-1, 1), name="batch_index")])

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)
