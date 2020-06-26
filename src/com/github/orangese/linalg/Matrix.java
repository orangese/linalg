package com.github.orangese.linalg;

import com.github.orangese.linalg.indexing.Slice;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Matrix extends LAObject {

    private final List<LAObject> data;

    public Matrix(Number ... data) throws IllegalArgumentException {
        this(Arrays.stream(data).map(Scalar::new).collect(Collectors.toList()));
    }

    public Matrix(LAObject ... data) throws IllegalArgumentException {
        this(Arrays.asList(data));
    }

    public Matrix(List<LAObject> data) {
        if (data.size() < 1) {
            throw new IllegalArgumentException("data must be provided");
        }

        this.data = data;
        this.setShape(new Dimension(data.get(0).shape(), data.size()));
    }

    public Matrix slice() {
        return new Matrix(data.subList(0, data.size()));
    }

    public LAObject slice(Slice slice) {
        int start = slice.calcNewStart(shape());
        int stop = slice.calcNewStop(shape());

        if (slice.isRange()) {
            slice.checkRange(stop, shape().getAlongAxis(0));
            List<LAObject> sliced = data.subList(start, stop);

            return sliced.size() == 1 ? sliced.get(0) : new Matrix(sliced);

        } else {
            LAObject currentSliced = this;
            for (Slice subSlice : slice.getSubSlices()) {
                currentSliced = currentSliced.slice(subSlice);
            }
            return currentSliced;
        }
    }

    public double prod(Matrix mat) throws IllegalArgumentException {
        if (shape().getDims().length > 2 || mat.shape().getDims().length > 2) {
            throw new IllegalArgumentException("only 2x2 or smaller prods supported");
        } if (shape().getAlongAxis(-1) != mat.shape().getAlongAxis(0)) {
            throw new IllegalArgumentException("prod does not exist between shapes " + shape() + " and " + mat.shape());
        }
        return -0.;
    }

    public String partialToString() {
        return toString() + "\n";
    }

    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (LAObject obj : data) {
            result.append(obj.partialToString());
        }
        result.deleteCharAt(result.length() - 1);
        return result.append("]").toString();
    }

}
