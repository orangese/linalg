package com.github.orangese.linalg;

import java.util.Arrays;
import java.util.Objects;

public abstract class LinAlgObj {

    private double[] data;
    private Shape shape;
    private int size;

    protected void setData(double[] data) {
        this.data = data;
    }

    public double[] data() {
        return data;
    }

    protected void setShape(Shape shape) {
        this.shape = shape;
        this.size = shape.size();
    }

    public Shape shape() {
        return shape;
    }

    public int ndims() {
        return shape().length;
    }

    public int size() {
        return size;
    }

    protected abstract void checkAddShapes(LinAlgObj other, String op);

    protected abstract void checkMulShapes(LinAlgObj other, String op);

    public abstract <T extends LinAlgObj> LinAlgObj add(T other);

    public abstract <T extends LinAlgObj> LinAlgObj subtract(T other);

    public abstract <T extends LinAlgObj> LinAlgObj mul(T other);

    public abstract LinAlgObj pow(Scalar scalar);

    public abstract <T extends LinAlgObj> void iadd(T other);

    public abstract <T extends LinAlgObj> void isubtract(T other);

    public abstract <T extends LinAlgObj> void imul(T other);

    public abstract void ipow(Scalar scalar);

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        } if (!(other instanceof LinAlgObj)) {
            return false;
        }
        LinAlgObj otherLinAlgObj = (LinAlgObj) other;
        return Arrays.equals(data, otherLinAlgObj.data) && shape.equals(otherLinAlgObj.shape);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

}
