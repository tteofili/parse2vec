package com.github.tteofili.parse2vec;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.charfilter.HTMLStripCharFilterFactory;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.core.TypeTokenFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.en.EnglishPossessiveFilterFactory;
import org.apache.lucene.analysis.opennlp.OpenNLPChunkerFilterFactory;
import org.apache.lucene.analysis.opennlp.OpenNLPPOSFilterFactory;
import org.apache.lucene.analysis.opennlp.OpenNLPTokenizerFactory;
import org.apache.lucene.analysis.pattern.PatternReplaceFilterFactory;
import org.apache.lucene.analysis.shingle.ShingleFilterFactory;
import org.apache.lucene.analysis.standard.ClassicTokenizerFactory;

/**
 *
 */
class AnalysisUtils {

  static Analyzer openNLPAnalyzer() throws Exception {
    String sentenceModel = "en-sent.bin";
    String tokenizerModel = "en-token.bin";
    return CustomAnalyzer.builder()
          .addCharFilter(HTMLStripCharFilterFactory.class)
          .withTokenizer(OpenNLPTokenizerFactory.class, OpenNLPTokenizerFactory.SENTENCE_MODEL,
              sentenceModel, OpenNLPTokenizerFactory.TOKENIZER_MODEL, tokenizerModel)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .build();

  }

  static Analyzer simpleAnalyzer() throws Exception {
    return CustomAnalyzer.builder()
        .addCharFilter(HTMLStripCharFilterFactory.class)
        .withTokenizer(ClassicTokenizerFactory.class)
        .addTokenFilter(EnglishPossessiveFilterFactory.class)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .build();
  }

  static Analyzer shingleSimpleAnalyzer() throws Exception {
    return CustomAnalyzer.builder()
        .addCharFilter(HTMLStripCharFilterFactory.class)
        .withTokenizer(ClassicTokenizerFactory.class)
        .addTokenFilter(EnglishPossessiveFilterFactory.class)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .addTokenFilter(ShingleFilterFactory.class, "minShingleSize", "2", "maxShingleSize",
            "3", "outputUnigrams", "true", "outputUnigramsIfNoShingles", "false", "tokenSeparator", " ")
        .build();
  }
}
