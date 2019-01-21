# network-embedding-evaluation

## Tentative Folder Structure:
- embeddings/\<embedding_method\>/ - holds all relevant exports for a given embedding method (HEER, DistMult, ComplEx, ConvE, etc.)

- datasets/FB15K237/ - holds all testing and training datasets as well as other FB15K237-specific metadata needed by the testing framework

- models/ - contains class definitions for each embedding method, these classes will just load existing embeddings from embeddings/ and node/edge IDs from datasets/ and execute prediction functions for testing

- evaluate.py - top-level evaluation script that performs all tests on all models and reports metrics

## Testing
To run the edge reconstruction experiment (modeled from HEER paper), just supply a model name as the first argument to the top-level script (i.e. `./eval.sh HEER` or `./eval.sh DistMult`). The results will be printed on screen.

Since some models use PyTorch and other use TensorFlow and these libraries require different Python versions (2 and 3 respectively), there are two prediction Python scripts `predict_head_tail.py` (runs PyTorch models like HEER) and `predict_head_tail_tf.py`(runs TensorFlow models like ProjE), and `eval.sh` will decide which one to run depending on the provided model name (please add your model to the appropriate array in the script).

## Notes
- This pipeline expects model classes to expose a `predict(head, tail, rel, pred_tail=True)` function where a single head, tail and relation value are provided. Depending on the underlying model implementation, these values may need to be transformed into vectors.
- To corrupt head or tail values during predictions, we simply hard-code the vector with `range(14505)` (for HEER) and `range(14541)` (for DistMult and ComplEx), which is just all the IDs of the nodes these models have seen. The fact that DistMult and ComplEx are at least aware of test-only nodes does not actually affect their performance much.
- Much of the dictionaries used to run these tests and represent the graph network of interest are based off of the HEER code, which adds a layer of complexity to handle multiple node types. They have been slightly simplified under the assumption that IDs correspond directly to their embedding index (i.e. the value used to read from the embedding matrix).
