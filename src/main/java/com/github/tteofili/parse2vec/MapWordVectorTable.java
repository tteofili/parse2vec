/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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