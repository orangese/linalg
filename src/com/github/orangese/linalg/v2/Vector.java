package com.github.orangese.linalg.v2;

public class Vector extends Matrix {

    public Vector(double[] data) {
        super(data);
    }

    public Vector(double[][] data) throws IllegalArgumentException {
        super(data);
        if (data.length != 1 && data[0].length != 1) {
            throw new IllegalArgumentException("Vector must be n x 1 or 1 x n");
        }
    }

    public double dot(Vector other) {
        checkAddShapes(other); // add shape requirements == dot prod shape requirements
        double prod = 0;
        for (int i = 0; i < data().length; i++) {
            prod += data()[i] * other.data()[i];
        }
        return prod;
    }

}
