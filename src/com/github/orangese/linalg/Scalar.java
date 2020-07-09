package com.github.orangese.linalg;

public class Scalar extends LinAlgObj {

    public Scalar(double val) {
        this.setData(new double[]{val});
        this.setShape(new int[]{});
    }

    public double val() {
        return data()[0];
    }

    @Override
    protected void checkAddShapes(LinAlgObj other) throws UnsupportedOperationException {
        if (other.ndims() != 0) {
            throw new UnsupportedOperationException("cannot add scalar and non-scalar with ndims " + other.ndims());
        }
    }

    @Override
    protected void checkMulShapes(LinAlgObj other) throws UnsupportedOperationException {
        if (other.ndims() != 0) {
            throw new UnsupportedOperationException("cannot mul scalar and non-scalar with ndims " + other.ndims());
        }
    }

    @Override
    public Scalar add(LinAlgObj other) {
        checkAddShapes(other);
        return new Scalar(val() + ((Scalar) other).val());
    }

    @Override
    public Scalar subtract(LinAlgObj other) {
        checkAddShapes(other);
        return new Scalar(val() + ((Scalar) other).val());
    }

    @Override
    public LinAlgObj mul(LinAlgObj other) {
        double[] newData = new double[other.data().length];
        for (int i = 0; i < other.data().length; i++) {
            newData[i] = other.data()[i] * val();
        }
        return newData.length > 1 ? new Matrix(newData, other.shape()) : new Scalar(newData[0]);
    }

    public Scalar div(Scalar other) {
        return new Scalar(val() / other.val());
    }

    @Override
    public void iadd(LinAlgObj other) {
        checkAddShapes(other);
        data()[0] += ((Scalar) other).val();
    }

    @Override
    public void isubtract(LinAlgObj other) {
        checkAddShapes(other);
        data()[0] -= ((Scalar) other).val();
    }

    public void imul(LinAlgObj other) {
        checkMulShapes(other);
        data()[0] *= ((Scalar) other).val();
    }

    public void idiv(Scalar other) {
        checkAddShapes(other);
        data()[0] /= other.val();
    }

    @Override
    public String toString() {
        return String.valueOf(this.val());
    }
}
