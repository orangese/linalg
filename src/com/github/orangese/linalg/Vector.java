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

    public double get(int idx) {
        return data()[idx];
    }

    public void set(int idx, double newVal) {
        data()[idx] = newVal;
    }

    public Vector add(Vector o) {
        return asVector(super.add(o));
    }

    public Vector subtract(Vector o) {
        return asVector(super.subtract(o));
    }

    public Vector mul(Vector o) {
        return asVector(super.mul(o));
    }

    @Override
    public Vector transpose() {
        return asVector(super.transpose());
    }

    public static Vector asVector(Matrix o) {
        Vector vec = new Vector();

        vec.setData(o.data());
        vec.setShape(o.shape());

        return vec;
    }

    public Scalar dot(Vector o) {
        checkAddShapes(o, "dot product"); // add shape requirements == dot prod shape requirements
        double prod = 0;
        for (int i = 0; i < data().length; i++) {
            prod += get(i) * o.get(i);
        }
        return new Scalar(prod);
    }

}
