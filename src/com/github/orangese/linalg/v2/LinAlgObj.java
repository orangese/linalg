package com.github.orangese.linalg.v2;

import java.util.Arrays;

public abstract class LinAlgObj {

    private double[] data;
    private int[] shape;

    protected void setData(double[] data) {
        this.data = data;
    }

    public double[] data() {
        return data;
    }

    protected void setShape(int[] shape) {
        this.shape = shape;
    }

    public int[] shape() {
        return shape;
    }

    public int ndims() {
        return shape().length;
    }

    protected abstract void checkAddShapes(LinAlgObj other) throws UnsupportedOperationException;

    protected abstract void checkMulShapes(LinAlgObj other) throws UnsupportedOperationException;

    public abstract LinAlgObj add(LinAlgObj other);

    public abstract LinAlgObj subtract(LinAlgObj other);

    public abstract LinAlgObj mul(LinAlgObj other);

    public abstract void iadd(LinAlgObj other);

    public abstract void isubtract(LinAlgObj other);

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{" +
                "data=" + Arrays.toString(data()) +
                ", shape=" + Arrays.toString(shape()) +
                '}';
    }

}
