package com.github.orangese.linalg;

import com.github.orangese.linalg.indexing.Slice;

public abstract class LAObject {

    private Dimension shape;

    public Dimension shape() {
        return shape;
    }

    protected void setShape(Dimension shape) {
        this.shape = shape;
    }

    abstract LAObject slice(Slice slice);

    abstract String partialToString();

}
