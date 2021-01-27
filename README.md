# parse2vec

Tool for generating parse tree embeddings, parse tree enriched word embeddings and parse tree enriched sentence embeddings.

## How it works

Given a set of sentences (one by line) in a text file, this tool:
* learns word embeddings using [word2vec](https://arxiv.org/abs/1310.4546)
* builds the parse tree of each sentence
* using the parse tree structure it recursively averages word embeddings *PoS Type-wise* from all the sentences' parse tree
* each PoS tag finally has an embedding
* to enrich the word embedding with parse tree information, for each existing word:
    * recursively sums the type embeddings of the word ancestors (in the parse tree) and averages the result with its word embedding
* to generate a sentence embedding enhanced with parse tree information:
    * recursively builds the sentence vector using parse tree enriched word embeddings using the algorithm from [par2hier](https://www.sciencedirect.com/science/article/pii/S1877050917306154) to build sentence vectors from hierarchical structures

## Examples    

![Parse Tree Embeddings as visualized in TensorBoard](src/test/resources/outputs/pt_tb_vis.png)

parse tree embeddings nearest neighbour sample results:

```
nearest(VB) = VP
nearest(JJR) = RBR
nearest(CONJP) = AUX
...
```

![Parse Tree Enriched Word Embeddings as visualized in TensorBoard](src/test/resources/outputs/ptwords_tb_vis.png)

parse tree enriched word embeddings sample results:

```
nearest(crowd) = multiple, man, ...
nearest(hierarchical) = relationship, soft-max ...
nearest(Sutskever) = Greg, Kai, ...
...
```
 
![Parse Tree Enriched Seentence Embeddings as visualized in TensorBoard](src/test/resources/outputs/ptsentences_tb_vis.png) 
 
parse tree enriched sentence embeddings sample results: 
```
nearest(In order to capture in a quantitative way the nuance necessary to distinguish man from woman ...) = 
 - In parallel in the last few years language models based on neural networks have been used to cope with complex natural language processing tasks like emotion and paraphrase detection.
 - Based on a recent work that proposed to learn a generic language model that can be modified through a set of document-specific parameters we explore use of new neural network models that are adapted to ad-hoc IR tasks.

nearest(We introduce a new dataset with human judgments on pairs of words in sentential context and ...) =
 - The result can be used to enrich lexicons of under-resourced languages to identify ambiguities and to perform clustering and classification .
 - We consider the conditional probabilities p(c|w) and given a corpus Text the goal is to set the parameters θ of p(c|w;θ) so as to maximize the corpus probability .
...
```