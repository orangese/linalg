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
    protected void checkAddShapes(LinAlgObj o, String op) {
        if (o.ndims() != 0) {
            throw new IllegalArgumentException(String.format(
                    "cannot perform %s on scalar and non-scalar with shape %s", op, o.shape()
            ));
        }
    }

    @Override
    protected void checkMulShapes(LinAlgObj o, String op) {
        checkAddShapes(o, op);
    }

    @Override
    public Scalar add(LinAlgObj o) {
        checkAddShapes(o, "scalar addition");
        return new Scalar(val() + ((Scalar) o).val());
    }

    @Override
    public Scalar subtract(LinAlgObj o) {
        checkAddShapes(o, "scalar subtraction");
        return new Scalar(val() + ((Scalar) o).val());
    }

    @Override
    public Matrix mul(Matrix o) {
        Matrix newMat = new Matrix(o.shape());
        for (int i = 0; i < o.data().length; i++) {
            newMat.data()[i] = val() * o.data()[i];
        }
        return newMat;
    }

    public Scalar mul(Scalar o) {
        return new Scalar(o.val() * val());
    }

    @Override
    public Scalar pow(Scalar scalar) {
        return new Scalar(Math.pow(val(), scalar.val()));
    }

    public Scalar div(Scalar o) {
        return new Scalar(val() / o.val());
    }

    @Override
    public void iadd(LinAlgObj o) {
        checkAddShapes(o, "scalar addition");
        set(val() + ((Scalar) o).val());
    }

    @Override
    public void isubtract(LinAlgObj o) {
        checkAddShapes(o, "scalar subtraction");
        set(val() - ((Scalar) o).val());
    }

    @Override
    public void imul(LinAlgObj o) {
        checkMulShapes(o, "scalar multiplication");
        set(val() * ((Scalar) o).val());
    }

    @Override
    public void ipow(Scalar scalar) {
        set(Math.pow(val(), scalar.val()));
    }

    public void idiv(Scalar o) {
        set(val() / o.val());
    }

    public Scalar add(double o) {
        return add(new Scalar(o));
    }

    public Scalar subtract(double o) {
        return add(new Scalar(o));
    }

    public Scalar mul(double o) {
        return mul(new Scalar(o));
    }

    public Scalar pow(double o) {
        return pow(new Scalar(o));
    }

    public Scalar div(double o) {
        return div(new Scalar(o));
    }

    public void iadd(double o) {
        iadd(new Scalar(o));
    }

    public void isubtract(double o) {
        isubtract(new Scalar(o));
    }

    public void imul(double o) {
        imul(new Scalar(o));
    }

    public void ipow(double o) {
        ipow(new Scalar(o));
    }

    public void idiv(double o) {
        idiv(new Scalar(o));
    }

    @Override
    public String toString() {
        return String.valueOf(this.val());
    }
}
