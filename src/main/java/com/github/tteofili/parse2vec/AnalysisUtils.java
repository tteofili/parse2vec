package com.github.tteofili.parse2vec;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.en.EnglishPossessiveFilterFactory;
import org.apache.lucene.analysis.opennlp.OpenNLPTokenizerFactory;
import org.apache.lucene.analysis.shingle.ShingleFilterFactory;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;

/**
 *
 */
class AnalysisUtils {

  static Analyzer openNLPAnalyzer() throws Exception {
    String sentenceModel = "en-sent.bin";
    String tokenizerModel = "en-token.bin";
    return CustomAnalyzer.builder()
          .withTokenizer(OpenNLPTokenizerFactory.class, OpenNLPTokenizerFactory.SENTENCE_MODEL,
              sentenceModel, OpenNLPTokenizerFactory.TOKENIZER_MODEL, tokenizerModel)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .build();

  }

  static Analyzer simpleAnalyzer() throws Exception {
    return CustomAnalyzer.builder()
        .withTokenizer(StandardTokenizerFactory.class)
        .addTokenFilter(EnglishPossessiveFilterFactory.class)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .build();
  }

  static Analyzer shingleSimpleAnalyzer() throws Exception {
    return CustomAnalyzer.builder()
        .withTokenizer(StandardTokenizerFactory.class)
        .addTokenFilter(EnglishPossessiveFilterFactory.class)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .addTokenFilter(ShingleFilterFactory.class, "minShingleSize", "2", "maxShingleSize",
            "3", "outputUnigrams", "true", "outputUnigramsIfNoShingles", "false", "tokenSeparator", " ")
        .build();
  }
}
