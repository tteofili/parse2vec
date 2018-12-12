package com.github.tteofili.parse2vec;

import opennlp.tools.util.wordvector.WordVector;
import opennlp.tools.util.wordvector.WordVectorType;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;

class FloatArrayVector implements WordVector {

  private final float[] vector;

  FloatArrayVector(float[] vector) {
    this.vector = vector;
  }

  @Override
  public WordVectorType getDataType() {
    return WordVectorType.FLOAT;
  }

  @Override
  public float getAsFloat(int index) {
    return vector[index];
  }

  @Override
  public double getAsDouble(int index) {
    return getAsFloat(index);
  }

  @Override
  public FloatBuffer toFloatBuffer() {
    return FloatBuffer.wrap(vector);
  }

  @Override
  public DoubleBuffer toDoubleBuffer() {
    double[] doubleVector = new double[vector.length];
    for (int i = 0; i < doubleVector.length ; i++) {
      doubleVector[i] = vector[i];
    }
    return DoubleBuffer.wrap(doubleVector);
  }

  @Override
  public int dimension() {
    return vector.length;
  }

  @Override
  public String toString() {
    return Arrays.toString(vector);
  }
}
