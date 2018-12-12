package com.github.tteofili.parse2vec;

import org.junit.Test;

import java.util.HashMap;

/**
 * Tests for {@link Parse2Vec}
 */
public class Parse2VecTest {

    @Test
    public void testTSVWrite() throws Exception {
        int decimals = 1;
        MapWordVectorTable wvt = new MapWordVectorTable(new HashMap<>());
        float[] floats = new float[3];
        floats[0] = 0.11111111f;
        floats[1] = 0.211112311f;
        floats[2] = 0.39112311f;
        wvt.put("foo", new FloatArrayVector(floats));

        float[] floats1 = new float[3];
        floats1[0] = 0.21312311f;
        floats1[1] = 0.2313141f;
        floats1[2] = 0.9131231f;
        wvt.put("bar", new FloatArrayVector(floats1));

        String prefix = "target/sample";
        Parse2Vec.writeEmbeddingsAsTSV(wvt, prefix, decimals);
    }

}