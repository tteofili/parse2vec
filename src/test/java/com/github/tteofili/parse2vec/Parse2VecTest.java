package com.github.tteofili.parse2vec;

import org.junit.Test;

/**
 * Tests for {@link Parse2Vec}
 */
public class Parse2VecTest {

    @Test
    public void testExecution() throws Exception {
        Parse2Vec.main(new String[]{"src/test/resources/test-text"});
    }

}