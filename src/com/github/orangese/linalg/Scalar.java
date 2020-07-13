package com.github.orangese.linalg;

public class Scalar extends LinAlgObj {

    public Scalar(double val) {
        this.setData(new double[]{val});
        this.setShape(new Shape());
    }

    public double val() {
        return data()[0];
    }

    public void set(double val) {
        this.setData(new double[]{val});
    }

    @Override
    protected void checkAddShapes(LinAlgObj other, String op) {
        if (other.ndims() != 0) {
            throw new IllegalArgumentException(String.format(
                    "cannot perform %s on scalar and non-scalar with shape %s", op, other.shape()
            ));
        }
    }

    @Override
    protected void checkMulShapes(LinAlgObj other, String op) {
        checkAddShapes(other, op);
    }

    @Override
    public <T extends LinAlgObj> Scalar add(T other) {
        checkAddShapes(other, "scalar addition");
        return new Scalar(val() + ((Scalar) other).val());
    }

    @Override
    public <T extends LinAlgObj> Scalar subtract(T other) {
        checkAddShapes(other, "scalar subtraction");
        return new Scalar(val() + ((Scalar) other).val());
    }

    @Override
    public <T extends LinAlgObj> T mul(T other) {
        double[] newData = new double[other.size()];
        for (int i = 0; i < other.size(); i++) {
            newData[i] = other.data()[i] * val();
        }
        return (T) (newData.length > 1 ? new Matrix(newData, other.shape()) : new Scalar(newData[0]));
    }

    @Override
    public Scalar pow(Scalar scalar) {
        return new Scalar(Math.pow(val(), scalar.val()));
    }

    public Scalar div(Scalar other) {
        return new Scalar(val() / other.val());
    }

    @Override
    public <T extends LinAlgObj> void iadd(T other) {
        checkAddShapes(other, "scalar addition");
        set(val() + ((Scalar) other).val());
    }

    @Override
    public <T extends LinAlgObj> void isubtract(T other) {
        checkAddShapes(other, "scalar subtraction");
        set(val() - ((Scalar) other).val());
    }

    @Override
    public <T extends LinAlgObj> void imul(T other) {
        checkMulShapes(other, "scalar multiplication");
        set(val() * ((Scalar) other).val());
    }

    @Override
    public void ipow(Scalar scalar) {
        set(Math.pow(val(), scalar.val()));
    }

    public void idiv(Scalar other) {
        set(val() / other.val());
    }

    public Scalar add(double other) {
        return add(new Scalar(other));
    }

    public Scalar subtract(double other) {
        return add(new Scalar(other));
    }

    public Scalar mul(double other) {
        return mul(new Scalar(other));
    }

    public Scalar pow(double other) {
        return pow(new Scalar(other));
    }

    public Scalar div(double other) {
        return div(new Scalar(other));
    }

    public void iadd(double other) {
        iadd(new Scalar(other));
    }

    public void isubtract(double other) {
        isubtract(new Scalar(other));
    }

    public void imul(double other) {
        imul(new Scalar(other));
    }

    public void ipow(double other) {
        ipow(new Scalar(other));
    }

    public void idiv(double other) {
        idiv(new Scalar(other));
    }

    protected static boolean isNonZero(double a) {
        return Math.abs(a) > Math.max(1e-09 * Math.abs(a), Matrix.EPS);
    }

    @Override
    public String toString() {
        return String.valueOf(this.val());
    }
}
