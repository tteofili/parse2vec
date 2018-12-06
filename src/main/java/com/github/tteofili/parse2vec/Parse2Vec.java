package com.github.tteofili.parse2vec;

import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.util.wordvector.WordVector;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Parse2Vec {

    private static Logger logger = LoggerFactory.getLogger(Parse2Vec.class);

    public static void main(String[] args) throws Exception {
        Path path = Paths.get("src/test/resources/text");

        InputStream sentenceModelStream = new FileInputStream("src/test/resources/en-sent.bin");
        InputStream parserModelStream = new FileInputStream("src/test/resources/en-parser-chunking.bin");
        try {

            // train word embeddings
            int layerSize = 100;
            Word2Vec word2Vec = new Word2Vec.Builder()
                    .tokenizerFactory(new DefaultTokenizerFactory())
                    .epochs(5)
                    .layerSize(layerSize)
                    .iterate(new FileDocumentIterator(path.toFile()))
                    .build();
            word2Vec.fit();

            SentenceModel sentenceModel = new SentenceModel(sentenceModelStream);
            SentenceDetectorME sentenceDetector = new SentenceDetectorME(sentenceModel);

            ParserModel parserModel = new ParserModel(parserModelStream);
            Parser parser = ParserFactory.create(parserModel);

            File dir = path.toFile();

            MapWordVectorTable ptEmbeddings = extractPTEmbeddings(layerSize, word2Vec, sentenceDetector, parser, dir);
            writeEmbeddingsAsTSV(ptEmbeddings, "pt", 1);

            MapWordVectorTable parsePathWordEmbeddings = extractPTPathWordEmbeddings(word2Vec, sentenceDetector, parser, dir, ptEmbeddings);
            writeEmbeddingsAsTSV(parsePathWordEmbeddings, "pt-word", 1);

        } finally {
            sentenceModelStream.close();
            parserModelStream.close();
        }
    }

    static void writeEmbeddingsAsTSV(MapWordVectorTable wordVectorTable, String prefix, int decimals) throws IOException {
        double rounding = Math.pow(10, decimals);
        Charset charset = Charset.forName("UTF-8");
        byte[] tabBytes = "\t".getBytes(charset);
        byte[] crBytes = "\n".getBytes(charset);

        FileOutputStream vectorsFileStream = new FileOutputStream(prefix + "-vectors.tsv");
        FileOutputStream metadataFileStream = new FileOutputStream(prefix + "-metadata.tsv");

        Iterator<String> tokens = wordVectorTable.tokens();
        try {
            while (tokens.hasNext()) {
                String pt = tokens.next();
                metadataFileStream.write(pt.getBytes(charset));
                metadataFileStream.write(crBytes);
                float[] array = wordVectorTable.get(pt).toFloatBuffer().array();
                for (float f : array) {
                    double v = Math.round(f * rounding) / rounding;
                    vectorsFileStream.write(String.valueOf(v).getBytes(charset));
                    vectorsFileStream.write(tabBytes);
                }
                vectorsFileStream.write(crBytes);
            }
        } finally {
            metadataFileStream.flush();
            metadataFileStream.close();

            vectorsFileStream.flush();
            vectorsFileStream.close();
        }
    }

    @NotNull
    private static MapWordVectorTable extractPTPathWordEmbeddings(Word2Vec word2Vec, SentenceDetectorME sentenceDetector, Parser parser, File dir, MapWordVectorTable ptEmbeddings) throws IOException {
        logger.info("extracting parse tree enriched word embeddings");
        MapWordVectorTable parsePathWordEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
                        for (Parse topParse : topParses) {
                            // exclude TOPs
                            for (Parse p : topParse.getChildren()) {
                                Parse[] tagNodes = p.getTagNodes();
                                // record pt path word embeddings for leaf nodes
                                for (Parse tn : tagNodes) {
                                    String word = tn.getCoveredText();
                                    if (word != null) {
                                        INDArray vector = word2Vec.getLookupTable().vector(word);
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
                }
            }
        }
        return parsePathWordEmbeddings;
    }

    @NotNull
    private static MapWordVectorTable extractPTEmbeddings(int layerSize, Word2Vec word2Vec, SentenceDetectorME sentenceDetector, Parser parser, File dir) throws IOException {
        logger.info("extracting parse tree embeddings");
        MapWordVectorTable ptEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    // todo : normalize text, eventually
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
                        for (Parse topParse : topParses) {
                            // exclude TOPs
                            for (Parse p : topParse.getChildren()) {
                                Parse[] tagNodes = p.getTagNodes();
                                // record pt embeddings for leaf nodes
                                for (Parse tn : tagNodes) {
                                    INDArray vector = word2Vec.getLookupTable().vector(tn.getCoveredText());
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
                }
            }
        }
        return ptEmbeddings;
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
