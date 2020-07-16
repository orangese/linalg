package com.github.orangese.linalg;

public class Vector extends Matrix {

    public Vector(Shape shape) {
        super(shape);
    }

    public Vector(Shape shape, double fillVal) {
        super(shape, fillVal);
    }

    public Vector(double... data) {
        super(data, new Shape(data.length, 1));
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
        setData(other.data());
        setShape(other.shape());
    }

    public double get(int idx) {
        return data()[idx];
    }

    public void set(int idx, double newVal) {
        data()[idx] = newVal;
    }

    public Vector add(Vector other) {
        return new Vector(super.add(other));
    }

    public Vector subtract(Vector other) {
        return new Vector(super.subtract(other));
    }

    public Vector mul(Vector other) {
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
            prod += get(i) * other.get(i);
        }
        return new Scalar(prod);
    }

}
