package com.github.tteofili.parse2vec;

import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;
import opennlp.tools.util.wordvector.WordVector;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Set;

import static com.github.tteofili.parse2vec.EmbeddingsUtils.getTruncatedVT;

class Parse2VecUtils {

    private static final Logger logger = LoggerFactory.getLogger(Parse2VecUtils.class);

    static void getPTPathSentenceEmbedding(Parser parser, MapWordVectorTable ptEmbeddings, MapWordVectorTable parsePathWordEmbeddings,
                                           int k, Parse2Vec.Method method, int layerSize, MapWordVectorTable parsePathSentenceEmbeddings,
                                           TokenizerFactory tokenizerFactory, String sentence) {
        Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
        String id = sentence.replaceAll(" ", "_").replaceAll("\t", "").replaceAll("\n", "").replaceAll("\r", "");
        for (Parse topParse : topParses) {
            INDArray sentenceVector = getSentenceVector(ptEmbeddings, parsePathWordEmbeddings,
                    topParse, k, layerSize, method, tokenizerFactory);
            parsePathSentenceEmbeddings.put(id, new FloatArrayVector(sentenceVector.toFloatVector()));
        }
    }

    private static INDArray getSentenceVector(MapWordVectorTable ptEmbeddings, MapWordVectorTable parsePathWordEmbeddings,
                                              Parse parseTree, int k, int layerSize, Parse2Vec.Method method, TokenizerFactory tokenizerFactory) {

        Parse[] children = parseTree.getChildren();
        if (children.length == 0) {
            String coveredText = parseTree.getCoveredText().trim();
            WordVector wordVector = parsePathWordEmbeddings.get(coveredText);
            INDArray vector = null;
            if (wordVector != null) {
                vector = Nd4j.create(wordVector.toFloatBuffer().array());
            } else {
                logger.warn("cannot find word vector for {}", coveredText);
                WordVector ptVector = ptEmbeddings.get(parseTree.getType());
                if (ptVector != null) {
                    vector = Nd4j.create(ptVector.toFloatBuffer().array());
                } else {
                    logger.warn("cannot find pt vector for {}", parseTree.getType());
                }
            }
            if (vector == null) {
                logger.error("cannot find vector for {}", parseTree.toString());
                vector = Nd4j.zeros(1, layerSize);
            }
            return vector;

        } else {
            INDArray chvs = Nd4j.zeros(children.length, layerSize);
            int i = 0;
            for (Parse desc : children) {
                INDArray chv = getSentenceVector(ptEmbeddings, parsePathWordEmbeddings, desc, k, layerSize, method, tokenizerFactory);
                chvs.putRow(i, chv);
                i++;
            }

            INDArray hv = Nd4j.create(ptEmbeddings.get(parseTree.getType()).toFloatBuffer().array());
            double[][] centroids;
            if (chvs.rows() > k) {
                centroids = EmbeddingsUtils.getTruncatedVT(chvs, k);
            } else if (chvs.rows() == 1) {
                centroids = EmbeddingsUtils.getDoubles(chvs.getRow(0));
            } else {
                centroids = getTruncatedVT(chvs, 1);
            }
            switch (method) {
                case CLUSTER:
                    INDArray matrix = Nd4j.zeros(centroids.length + 1, layerSize);
                    matrix.putRow(0, hv);
                    for (int c = 0; c < centroids.length; c++) {
                        matrix.putRow(c + 1, Nd4j.create(centroids[c]));
                    }
                    hv = Nd4j.create(getTruncatedVT(matrix, 1));
                    break;
                case SUM:
                    for (double[] centroid : centroids) {
                        hv.addi(Nd4j.create(centroid));
                    }
                    break;
            }

            return hv;
        }
    }

    static void getPTPathWordEmbeddings(WordVectors wordVectors, Parser parser, MapWordVectorTable ptEmbeddings,
                                        MapWordVectorTable parsePathWordEmbeddings, String sentence) {
        Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
        for (Parse topParse : topParses) {
            // exclude TOPs
            for (Parse p : topParse.getChildren()) {
                Parse[] tagNodes = p.getTagNodes();
                // record pt path word embeddings for leaf nodes
                for (Parse tn : tagNodes) {
                    String word = tn.getCoveredText();
                    if (word != null) {
                        INDArray vector = wordVectors.getWordVectorMatrix(word);
                        if (vector != null) {
                            INDArray originalWordVector = vector.dup();
                            Parse parent;
                            while ((parent = tn.getParent()) != null) {
                                WordVector wordVector = ptEmbeddings.get(parent.getType());
                                originalWordVector.addi(Nd4j.create(wordVector.toFloatBuffer().array()));
                                tn = parent;
                            }
                            WordVector existingWordVector = parsePathWordEmbeddings.get(word);
                            if (existingWordVector != null) {
                                INDArray[] ind = new INDArray[2];
                                ind[0] = Nd4j.create(existingWordVector.toFloatBuffer().array());
                                ind[1] = originalWordVector;
                                parsePathWordEmbeddings.put(word, new FloatArrayVector(Nd4j.averageAndPropagate(ind).data().asFloat()));
                            } else {
                                parsePathWordEmbeddings.put(word, new FloatArrayVector(originalWordVector.data().asFloat()));
                            }
                        }
                    }
                }
            }
        }
    }

    static void getPTEmbeddingFromSentence(int layerSize, WordVectors wordVectors, Parser parser, MapWordVectorTable ptEmbeddings, String sentence) {
        Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
        for (Parse topParse : topParses) {
            // exclude TOPs
            for (Parse p : topParse.getChildren()) {
                Parse[] tagNodes = p.getTagNodes();
                // record pt embeddings for leaf nodes
                for (Parse tn : tagNodes) {
                    INDArray vector = wordVectors.getWordVectorMatrix(tn.getCoveredText());
                    if (vector != null) {
                        String type = tn.getType();
                        if (type != null && type.trim().length() > 0) {
                            if (ptEmbeddings.get(type) == null) {
                                ptEmbeddings.put(type, new FloatArrayVector(vector.data().asFloat()));
                            } else {
                                INDArray[] ar = new INDArray[2];
                                WordVector wordVector = ptEmbeddings.get(type);
                                ar[0] = Nd4j.create(wordVector.toFloatBuffer().array());
                                ar[1] = vector.dup();
                                ptEmbeddings.put(type, new FloatArrayVector(Nd4j.averageAndPropagate(ar).data().asFloat()));
                            }
                        }
                    }
                }
                // all leaf pt embeddings have been added

                // recurse bottom up until TOP
                getPTEmbeddings(ptEmbeddings, layerSize, tagNodes);
            }

        }
    }

    private static void getPTEmbeddings(MapWordVectorTable ptEmbeddings, int layerSize, Parse[] parses) {
        Set<Parse> parents = new HashSet<>();
        for (Parse t : parses) {
            Parse parent = t.getParent();
            if (parent != null) {
                parents.add(parent);
            }
        }
        for (Parse parent : parents) {
            String type = parent.getType();
            if (type != null && type.trim().length() > 0) {
                Parse[] children = parent.getChildren();
                WordVector existingParentVector = ptEmbeddings.get(parent.getType());
                INDArray[] ar = new INDArray[children.length + (existingParentVector != null ? 1 : 0)];
                int i = 0;
                for (Parse child : children) {
                    String childType = child.getType();
                    if (childType != null && childType.trim().length() > 0) {
                        WordVector childVector = ptEmbeddings.get(childType);
                        if (childVector != null) {
                            INDArray vector = Nd4j.create(childVector.toFloatBuffer().array());
                            ar[i] = vector != null ? vector : Nd4j.zeros(1, layerSize);
                        } else {
                            ar[i] = Nd4j.zeros(1, layerSize);
                        }
                    } else {
                        ar[i] = Nd4j.zeros(1, layerSize);
                    }
                    i++;
                }
                if (existingParentVector != null) {
                    ar[children.length] = Nd4j.create(existingParentVector.toFloatBuffer().array());
                }
                ptEmbeddings.put(parent.getType(), new FloatArrayVector(Nd4j.averageAndPropagate(ar).data().asFloat()));
            }
        }
        if (!parents.isEmpty()) {
            getPTEmbeddings(ptEmbeddings, layerSize, parents.toArray(new Parse[0]));
        }
    }
}
