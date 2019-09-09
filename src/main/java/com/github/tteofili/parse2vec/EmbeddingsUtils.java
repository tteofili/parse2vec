package com.github.tteofili.parse2vec;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.Iterator;

class EmbeddingsUtils {

    static void writeEmbeddingsAsSSV(MapWordVectorTable wordVectorTable, String prefix, int decimals) throws IOException {
        double rounding = Math.pow(10, decimals);
        Charset charset = Charset.forName("UTF-8");
        byte[] spaceBytes = "\t".getBytes(charset);
        byte[] crBytes = "\n".getBytes(charset);

        FileOutputStream vectorsFileStream = new FileOutputStream(prefix + "-vectors.txt");
        Iterator<String> tokens = wordVectorTable.tokens();
        try {
            while (tokens.hasNext()) {
                String pt = tokens.next();
                vectorsFileStream.write(pt.getBytes(charset));
                vectorsFileStream.write(spaceBytes);
                float[] array = wordVectorTable.get(pt).toFloatBuffer().array();
                for (float f : array) {
                    double v = Math.round(f * rounding) / rounding;
                    vectorsFileStream.write(String.valueOf(v).getBytes(charset));
                    vectorsFileStream.write(spaceBytes);
                }
                vectorsFileStream.write(crBytes);
            }
        } finally {
            vectorsFileStream.flush();
            vectorsFileStream.close();
        }
    }

    static void writeEmbeddingsAsSSV(WeightLookupTable wordVectorTable, String prefix, int decimals) throws IOException {
        double rounding = Math.pow(10, decimals);
        Charset charset = Charset.forName("UTF-8");
        byte[] spaceBytes = "\t".getBytes(charset);
        byte[] crBytes = "\n".getBytes(charset);

        FileOutputStream vectorsFileStream = new FileOutputStream(prefix + "-vectors.txt");
        Collection<VocabWord> tokens = wordVectorTable.getVocabCache().tokens();
        try {
            for (VocabWord token : tokens) {
                vectorsFileStream.write(token.getWord().getBytes(charset));
                vectorsFileStream.write(spaceBytes);
                float[] array = wordVectorTable.vector(token.getWord()).toFloatVector();
                for (float f : array) {
                    double v = Math.round(f * rounding) / rounding;
                    vectorsFileStream.write(String.valueOf(v).getBytes(charset));
                    vectorsFileStream.write(spaceBytes);
                }
                vectorsFileStream.write(crBytes);
            }
        } finally {
            vectorsFileStream.flush();
            vectorsFileStream.close();
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

    static double[][] getTruncatedVT(INDArray matrix, int k) {
        double[][] data = getDoubles(matrix);

        SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(data));

        double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
        svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);
        return truncatedVT;
    }

    static double[][] getDoubles(INDArray matrix) {
        double[][] data = new double[matrix.rows()][matrix.columns()];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                data[i][j] = matrix.getDouble(i, j);
            }
        }
        return data;
    }
}
