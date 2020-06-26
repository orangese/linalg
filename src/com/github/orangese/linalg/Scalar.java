package com.github.orangese.linalg;

import com.github.orangese.linalg.indexing.Slice;

public class Scalar extends LAObject {

    private Number scalar;

    public Scalar(Number scalar) {
        this.scalar = scalar;
        this.setShape(new Dimension());
    }

    public Number value() {
        return scalar;
    }

    public void set(Number newScalar) {
        scalar = newScalar;
    }

    public LAObject slice(Slice slice) throws IllegalArgumentException {
        throw new IllegalArgumentException("Scalar cannot be sliced");
    }

    public String partialToString() {
        return " " + toString() + " ";
    }

    public String toString() {
        return String.valueOf(scalar);
    }
}
