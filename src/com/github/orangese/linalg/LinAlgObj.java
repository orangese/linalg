package com.github.orangese.linalg;

import java.util.Arrays;
import java.util.Objects;

public abstract class LinAlgObj {

    private double[] data;
    private Shape shape;

    protected void setData(double[] data) {
        this.data = data;
    }

    public double[] data() {
        return data;
    }

    protected void setShape(Shape shape) {
        this.shape = shape;
    }

    public Shape shape() {
        return shape;
    }

    public int rowDim() {
        return shape.rowDim();
    }

    public int colDim() {
        return shape.colDim();
    }

    public int ndims() {
        return shape.ndims();
    }

    public int size() {
        return shape.size();
    }

    protected abstract void checkAddShapes(LinAlgObj o, String op);

    protected abstract void checkMulShapes(LinAlgObj o, String op);

    public abstract LinAlgObj add(LinAlgObj o);

    public abstract LinAlgObj subtract(LinAlgObj o);

    public abstract LinAlgObj mul(Matrix o);

    public abstract LinAlgObj pow(Scalar scalar);

    public abstract void iadd(LinAlgObj o);

    public abstract void isubtract(LinAlgObj o);

    public abstract void imul(LinAlgObj o);

    public abstract void ipow(Scalar scalar);

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LinAlgObj linAlgObj = (LinAlgObj) o;
        return Arrays.equals(data, linAlgObj.data) &&
                Objects.equals(shape, linAlgObj.shape);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

}
