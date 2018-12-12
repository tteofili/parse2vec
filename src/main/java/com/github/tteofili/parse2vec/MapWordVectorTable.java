package com.github.tteofili.parse2vec;

import opennlp.tools.util.wordvector.WordVector;
import opennlp.tools.util.wordvector.WordVectorTable;

import java.util.Iterator;
import java.util.Map;

class MapWordVectorTable implements WordVectorTable {

  private final Map<String, WordVector> vectors;

  MapWordVectorTable(Map<String, WordVector> vectors) {
    this.vectors = vectors;
  }

  public void put(String token, WordVector wordVector) {
    vectors.put(token, wordVector);
  }

  @Override
  public WordVector get(String token) {
    return vectors.get(token);
  }

  @Override
  public Iterator<String> tokens() {
    return vectors.keySet().iterator();
  }

  @Override
  public int size() {
    return vectors.size();
  }

  @Override
  public int dimension() {
    if (vectors.size() > 0) {
      return vectors.values().iterator().next().dimension();
    }
    else {
      return -1;
    }
  }

  @Override
  public String toString() {
    return "{" + vectors + '}';
  }
}
