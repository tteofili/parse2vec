package com.github.tteofili.parse2vec;

import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
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
import java.util.HashMap;

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

            if (dir.listFiles() != null) { // thanks Francesco
                MapWordVectorTable ptEmbeddings = extractPTEmbeddings(layerSize, word2Vec, sentenceDetector, parser, dir);
                EmbeddingsUtils.writeEmbeddingsAsTSV(ptEmbeddings, "pt", 1);

                MapWordVectorTable parsePathWordEmbeddings = extractPTPathWordEmbeddings(word2Vec, sentenceDetector, parser,
                        dir, ptEmbeddings);
                EmbeddingsUtils.writeEmbeddingsAsTSV(parsePathWordEmbeddings, "pt-word", 1);

                MapWordVectorTable parsePathSentenceEmbeddings = extractPTPathSentenceEmbeddings(sentenceDetector,
                        parser, dir, ptEmbeddings, parsePathWordEmbeddings, 3, Method.CLUSTER, layerSize, tokenizerFactory);
                EmbeddingsUtils.writeEmbeddingsAsTSV(parsePathSentenceEmbeddings, "pt-sentence", 1);
            }

        } finally {
            sentenceModelStream.close();
            parserModelStream.close();
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
    private static MapWordVectorTable extractPTPathWordEmbeddings(WordVectors word2Vec, SentenceDetectorME sentenceDetector, Parser parser, File dir, MapWordVectorTable ptEmbeddings) throws IOException {
        logger.info("extracting parse tree enriched word embeddings");
        MapWordVectorTable parsePathWordEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse2VecUtils.getPTPathWordEmbeddings(word2Vec, parser, ptEmbeddings, parsePathWordEmbeddings, sentence);
                    }
                }
            }
        }
        return parsePathWordEmbeddings;
    }

    @NotNull
    private static MapWordVectorTable extractPTEmbeddings(int layerSize, WordVectors wordVectors, SentenceDetectorME sentenceDetector, Parser parser, File dir) throws IOException {
        logger.info("extracting parse tree embeddings");
        MapWordVectorTable ptEmbeddings = new MapWordVectorTable(new HashMap<>());
        for (File f : dir.listFiles()) {
            if (!f.getName().startsWith(".")) {
                logger.info("processing file {}", f);
                for (String line : IOUtils.readLines(new FileInputStream(f), Charset.defaultCharset())) {
                    // todo : normalize text, eventually
                    String[] sentences = sentenceDetector.sentDetect(line);
                    for (String sentence : sentences) {
                        Parse2VecUtils.getPTEmbeddingsFromSentence(layerSize, wordVectors, parser, ptEmbeddings, sentence);
                    }
                }
            }
        }
        return ptEmbeddings;
    }

}
