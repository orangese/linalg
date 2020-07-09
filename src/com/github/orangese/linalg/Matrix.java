package com.github.orangese.linalg;

import java.util.Arrays;

public class Matrix extends LinAlgObj {

    private int[] strides;

    public Matrix(double[] data) {
        this.setData(new double[data.length]);
        System.arraycopy(data, 0, this.data(), 0, data.length);
        this.setShape(new int[]{1, data.length});
    }

    public Matrix(double[] data, int[] shape) {
        this(data);
        System.arraycopy(shape, 0, this.shape(), 0, shape.length);
        this.setShape(shape());
    }

    public Matrix(double[][] data) throws IllegalArgumentException {
        this.setData(new double[data.length * data[0].length]);
        for (int axis = 0; axis < data.length; axis++) {
            if (data[axis].length != data[0].length) {
                throw new IllegalArgumentException(String.format(
                        "dim %d along axis %d != dim %d along axis 0",
                        data[axis].length, axis, data[0].length
                ));
            }
            System.arraycopy(data[axis], 0, this.data(), data[axis].length * axis, data[axis].length);
        }
        this.setShape(new int[]{data.length, this.data().length / data.length});
    }

    public Matrix(Matrix other) {
        setData(new double[other.data().length]);
        System.arraycopy(other.data(), 0, data(), 0, other.data().length);

        setShape(new int[other.ndims()]);
        System.arraycopy(other.shape(), 0, shape(), 0, other.ndims());

        strides = new int[other.strides.length];
        System.arraycopy(other.strides, 0, strides, 0, other.strides.length);
    }

    private Matrix(double[] data, int[] shape, int[] strides) {
        // used internally to create transposes, which are views
        this.setData(data);
        this.setShape(shape);
        this.strides = strides;
    }

    protected void setShape(int[] shape) {
        // strides/indexing written generally in case tensor support is added later
        strides = new int[shape.length];
        int currStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (i != shape.length - 1) {
                currStride *= shape[i + 1];
            }
            this.strides[i] = currStride;
        }
        super.setShape(shape);
    }

    public int[] strides() {
        return strides;
    }

    private int getStrided(int... idxs) throws IllegalArgumentException {
        if (idxs.length == 0) {
            throw new IllegalArgumentException("idxs must contain at least one idx");
        } else if (idxs.length != ndims()) {
            throw new IllegalArgumentException("indices length does not match ndims");
        }

        int strided = 0;
        for (int axis = 0; axis < ndims(); axis++) {
            int currIdx = idxs[axis];

            if (currIdx < -shape()[axis] || currIdx >= shape()[axis]) {
                throw new ArrayIndexOutOfBoundsException(String.format(
                        "index %d out of range for axis %d of length %d", currIdx, axis, shape()[axis]
                ));
            } if (currIdx < 0) {
                // negative indexes are supported
                currIdx += shape()[axis];
            }
            strided += currIdx * strides[axis];
        }
        return strided;
    }

    public double get(int... idxs) {
        return data()[getStrided(idxs)];
    }

    public void set(int[] idxs, double newVal) {
        data()[getStrided(idxs)] = newVal;
    }

    private void shapeFailure(LinAlgObj other) throws UnsupportedOperationException {
        throw new UnsupportedOperationException(String.format(
                "shape %s does not match shape %s for requested operation",
                Arrays.toString(shape()),
                Arrays.toString(other.shape())
        ));
    }

    protected void checkAddShapes(LinAlgObj other) {
        if (!Arrays.equals(shape(), other.shape())) {
            shapeFailure(other);
        }
    }

    protected void checkMulShapes(LinAlgObj other) {
        if (other.ndims() > 0 && shape()[ndims() - 1] != other.shape()[0]) {
            shapeFailure(other);
        }
    }

    private void imatMul(Matrix b, Matrix newMatrix) {
        for (int i = 0; i < shape()[0]; i++) {
            for (int j = 0; j < b.shape()[1]; j++) {
                for (int k = 0; k < shape()[1]; k++) {
                    int[] idxs = new int[]{i, j};
                    newMatrix.set(idxs, newMatrix.get(idxs) + get(i, k) * b.get(k, j));
                }
            }
        }
    }

    private void imatPow(Scalar scalar, Matrix newMatrix) {
        ;
    }

    @Override
    public <T extends LinAlgObj> Matrix add(T other) {
        checkAddShapes(other);
        double[] newData = new double[data().length];
        for (int i = 0; i < data().length; i++) {
            newData[i] = data()[i] + other.data()[i];
        }
        return new Matrix(newData, shape());
    }

    @Override
    public <T extends LinAlgObj> Matrix subtract(T other) {
        checkAddShapes(other);
        double[] newData = new double[data().length];
        for (int i = 0; i < data().length; i++) {
            newData[i] = data()[i] - other.data()[i];
        }
        return new Matrix(newData, shape());
    }

    @Override
    public <T extends LinAlgObj> LinAlgObj mul(T other) {
        checkMulShapes(other);
        if (other.ndims() == 0) {
            // scalar multiplication is communative
            return other.mul(this);
        } else {
            Matrix newMatrix = new Matrix(new double[shape()[0]][other.shape()[other.ndims() - 1]]);
            imatMul((Matrix) other, newMatrix);
            return newMatrix;
        }
    }

    @Override
    public LinAlgObj pow(Scalar scalar) {
        Matrix newMatrix = new Matrix(new double[data().length]);
        imatPow(scalar, newMatrix);
        return newMatrix;
    }

    @Override
    public <T extends LinAlgObj> void iadd(T other) {
        checkAddShapes(other);
        for (int i = 0; i < data().length; i++) {
            data()[i] += other.data()[i];
        }
    }

    @Override
    public <T extends LinAlgObj> void isubtract(T other) {
        checkAddShapes(other);
        for (int i = 0; i < data().length; i++) {
            data()[i] -= other.data()[i];
        }
    }

    @Override
    public <T extends LinAlgObj> void imul(T other) {
        checkAddShapes(other);
        imatMul((Matrix) other, this);
    }

    @Override
    public void ipow(Scalar scalar) {
        checkAddShapes(this);
        imatPow(scalar, this);
    }

    public Matrix transpose() {
        int[] newShape = new int[ndims()];
        int[] newStrides = new int[ndims()];
        for (int axis = 0; axis < ndims(); axis++) {
            newShape[axis] = shape()[ndims() - axis - 1];
            newStrides[axis] = strides[ndims() - axis - 1];
        }
        return new Matrix(data(), newShape, newStrides);
    }

    public Scalar det() {
        if (Arrays.equals(shape(), new int[]{2, 2})) {
            return new Scalar(data()[0] * data()[3] - data()[1] * data()[2]);
        } else {
            return null;
        }
    }

    public Matrix inv() {
        if (Arrays.equals(shape(), new int[]{2, 2})) {
            return det().pow(-1).mul(new Matrix(new double[][]{
                    new double[]{data()[3], -data()[1]},
                    new double[]{-data()[2], data()[0]}
            }));
        } else {
            return null;
        }
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (int i = 0; i < shape()[0]; i++) {
            for (int j = 0; j < shape()[1]; j++) {
                result.append(" ").append(get(i, j)).append(" ");
            }
            result.append("\n");
        }
        result.setLength(result.length() - 1);
        return result.append("]").toString();
    }

}
