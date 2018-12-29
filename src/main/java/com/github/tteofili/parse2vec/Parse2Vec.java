package com.github.tteofili.parse2vec;

import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.wordvector.WordVector;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

/**
 * Basic runner for pars2vec over a directory of text files
 */
public class Parse2Vec {

    enum Method {
        CLUSTER,
        SUM
    }

    private static Logger logger = LoggerFactory.getLogger(Parse2Vec.class);

    public static void main(String[] args) throws Exception {

        if (args.length == 0) {
            throw new Exception("please privide an input text to train the parse2vec models");
        }

        Path path = Paths.get(args[0]);

        InputStream sentenceModelStream = new FileInputStream("src/main/resources/en-sent.bin");
        InputStream parserModelStream = new FileInputStream("src/main/resources/en-parser-chunking.bin");
        try {

            // train word embeddings
            int layerSize = args.length > 1 && args[1] != null ? Integer.parseInt(args[1]) : 100;
            LuceneTokenizerFactory tokenizerFactory = new LuceneTokenizerFactory();
            Word2Vec word2Vec = new Word2Vec.Builder()
                    .tokenizerFactory(tokenizerFactory)
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

            if (dir.listFiles() != null) {
                MapWordVectorTable ptEmbeddings = extractPTEmbeddings(layerSize, word2Vec, sentenceDetector, parser, dir, tokenizerFactory);
                checkEmbeddings(ptEmbeddings, layerSize);
                EmbeddingsUtils.writeEmbeddingsAsTSV(ptEmbeddings, "pt-tags", 1);

                MapWordVectorTable parsePathWordEmbeddings = extractPTPathWordEmbeddings(word2Vec, sentenceDetector, parser,
                        dir, ptEmbeddings, tokenizerFactory);
                checkEmbeddings(ptEmbeddings, layerSize);
                checkEmbeddings(parsePathWordEmbeddings, layerSize);
                EmbeddingsUtils.writeEmbeddingsAsTSV(parsePathWordEmbeddings, "pt-word", 1);

                MapWordVectorTable parsePathSentenceEmbeddings = extractPTPathSentenceEmbeddings(sentenceDetector,
                        parser, dir, ptEmbeddings, parsePathWordEmbeddings, 3, Method.CLUSTER, layerSize, tokenizerFactory);
                checkEmbeddings(ptEmbeddings, layerSize);
                checkEmbeddings(parsePathWordEmbeddings, layerSize);
                checkEmbeddings(parsePathSentenceEmbeddings, layerSize);
                EmbeddingsUtils.writeEmbeddingsAsTSV(parsePathSentenceEmbeddings, "pt-sentence", 1);
            }

        } finally {
            sentenceModelStream.close();
            parserModelStream.close();
        }
    }

    private static void checkEmbeddings(MapWordVectorTable embeddings, int layerSize) {
        Iterator<String> tokens = embeddings.tokens();
        while (tokens.hasNext()) {
            String next = tokens.next();
            WordVector wordVector = embeddings.get(next);
            assert wordVector.dimension() == layerSize;
            if (wordVector.dimension() != layerSize) {
                logger.error("wrong dimension {} for token {} instead of {}", wordVector.dimension(), next, layerSize);
                float[] vector = new float[layerSize];
                Arrays.fill(vector, 1e-10f);
                embeddings.put(next, new FloatArrayVector(vector));
            }
        }
    }

    private static MapWordVectorTable extractPTPathSentenceEmbeddings(SentenceDetectorME sentenceDetector,
                                                                      Parser parser, File dir, MapWordVectorTable ptEmbeddings,
                                                                      MapWordVectorTable parsePathWordEmbeddings,
                                                                      int k, Method method, int layerSize, TokenizerFactory tokenizerFactory) throws IOException {
        logger.info("extracting parse tree enriched sentence embeddings");
        MapWordVectorTable parsePathSentenceEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse2VecUtils.getPTPathSentenceEmbedding(parser, ptEmbeddings, parsePathWordEmbeddings, k,
                                method, layerSize, parsePathSentenceEmbeddings, tokenizerFactory, sentence);
                    }
                }
            }
        }
        return parsePathSentenceEmbeddings;
    }

    @NotNull
    private static MapWordVectorTable extractPTPathWordEmbeddings(WordVectors wordVectors, SentenceDetectorME sentenceDetector,
                                                                  Parser parser, File dir, MapWordVectorTable ptEmbeddings,
                                                                  TokenizerFactory tokenizerFactory) throws IOException {
        logger.info("extracting parse tree enriched word embeddings");
        MapWordVectorTable parsePathWordEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse2VecUtils.getPTPathWordEmbeddings(wordVectors, parser, ptEmbeddings, parsePathWordEmbeddings, sentence, tokenizerFactory);
                    }
                }
            }
        }
        return parsePathWordEmbeddings;
    }

    @NotNull
    private static MapWordVectorTable extractPTEmbeddings(int layerSize, WordVectors wordVectors, SentenceDetectorME sentenceDetector,
                                                          Parser parser, File dir, TokenizerFactory tokenizerFactory) throws IOException {
        logger.info("extracting parse tree embeddings");
        MapWordVectorTable ptEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    // todo : normalize text, eventually
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse2VecUtils.getPTEmbeddingsFromSentence(layerSize, wordVectors, parser, ptEmbeddings, sentence, tokenizerFactory);
                    }
                }
            }
        }
        return ptEmbeddings;
    }

}
