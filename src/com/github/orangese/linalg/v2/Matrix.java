package com.github.orangese.linalg.v2;

import java.util.Arrays;

public class Matrix {

    private double[] data;
    private int[] shape;

    public Matrix(double[] data) {
        this.data = data;
        this.shape = new int[]{data.length};
    }

    public Matrix(double[][] data) throws IllegalArgumentException {
        this.data = new double[data.length * data[0].length];
        for (int axis = 0; axis < data.length; axis++) {
            if (data[axis].length != data[0].length) {
                throw new IllegalArgumentException(String.format(
                        "dim %d along axis %d != dim %d along axis 0",
                        data[axis].length, axis, data[0].length
                ));
            }
            System.arraycopy(data[axis], 0, this.data, data[axis].length * axis, data[axis].length);
        }
        this.shape = new int[]{data.length, this.data.length / data.length};
    }

    public double get(int ... idxs) throws IllegalArgumentException, ArrayIndexOutOfBoundsException {
        if (idxs.length == 0) {
            throw new IllegalArgumentException("idxs must contain at least one idx");
        } else if (idxs.length != shape.length) {
            throw new IllegalArgumentException("indices length does not match ndims");
        }

        int strided = 0;
        for (int axis = 0; axis < idxs.length; axis++) {
            int currIdx = idxs[axis];

            if (currIdx < 0) {
                currIdx += shape[axis];
            }

            if (currIdx >= shape[axis]) {
                throw new ArrayIndexOutOfBoundsException(String.format(
                        "index %d out of range for axis %d of length %d", currIdx, axis, shape[axis]
                ));
            } else if (axis == idxs.length - 1) {
                strided += currIdx;
            } else {
                strided += shape[axis + 1] * currIdx;
            }
        }
        return data[strided];
    }

    public int[] shape() {
        return shape;
    }

    public int ndims() {
        return shape.length;
    }

    @Override
    public String toString() {
        return "Matrix{" +
                "data=" + Arrays.toString(data) +
                ", shape=" + Arrays.toString(shape) +
                '}';
    }
}
