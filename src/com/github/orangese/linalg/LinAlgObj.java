package com.github.orangese.linalg;

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

    public abstract <T extends LinAlgObj> LinAlgObj add(T other);

    public abstract <T extends LinAlgObj> LinAlgObj subtract(T other);

    public abstract <T extends LinAlgObj> LinAlgObj mul(T other);

    public abstract LinAlgObj pow(Scalar scalar);

    public abstract <T extends LinAlgObj> void iadd(T other);

    public abstract <T extends LinAlgObj> void isubtract(T other);

    public abstract <T extends LinAlgObj> void imul(T other);

    public abstract void ipow(Scalar scalar);

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{" +
                "data=" + Arrays.toString(data()) +
                ", shape=" + Arrays.toString(shape()) +
                '}';
    }

}
