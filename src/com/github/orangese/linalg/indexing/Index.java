package com.github.orangese.linalg.indexing;

public class Index extends Slice {

    public Index(int idx) {
        super(idx, idx + 1);
    }

}
