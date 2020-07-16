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

    protected abstract void checkAddShapes(LinAlgObj other, String op);

    protected abstract void checkMulShapes(LinAlgObj other, String op);

    public abstract LinAlgObj add(LinAlgObj other);

    public abstract LinAlgObj subtract(LinAlgObj other);

    public abstract LinAlgObj mul(Matrix other);

    public abstract LinAlgObj pow(Scalar scalar);

    public abstract void iadd(LinAlgObj other);

    public abstract void isubtract(LinAlgObj other);

    public abstract void imul(LinAlgObj other);

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
