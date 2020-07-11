package com.github.orangese.linalg;

public class Vector extends Matrix {

    public Vector(double... data) {
        super(data);
    }

    public Vector(double[] data, Shape shape) {
        super(data, shape);
    }

    public Vector(double[][] data) {
        super(data);
        if (data.length != 1 && data[0].length != 1) {
            throw new IllegalArgumentException("Vector must be n x 1 or 1 x n");
        }
    }

    public Vector(Matrix other) {
        super(other);
    }

    @Override
    public <T extends LinAlgObj> Matrix add(T other) {
        return new Vector(super.add(other));
    }

    @Override
    public <T extends LinAlgObj> Matrix subtract(T other) {
        return new Vector(super.subtract(other));
    }

    @Override
    public <T extends LinAlgObj> Matrix mul(T other) {
        return new Vector(super.mul(other));
    }

    @Override
    public Vector transpose() {
        return new Vector(super.transpose());
    }

    public Scalar dot(Vector other) {
        checkAddShapes(other, "dot product"); // add shape requirements == dot prod shape requirements
        double prod = 0;
        for (int i = 0; i < data().length; i++) {
            prod += data()[i] * other.data()[i];
        }
        return new Scalar(prod);
    }

}
