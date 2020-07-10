package com.github.orangese.linalg;

public class Matrix extends LinAlgObj {

    public static int PRINT_PRECISION = 3;
    private int[] strides;

    public Matrix(double... data) {
        this.setData(new double[data.length]);
        System.arraycopy(data, 0, this.data(), 0, data.length);
        this.setShape(new Shape(1, data.length));
    }

    public Matrix(double[] data, Shape shape) {
        this(data);
        System.arraycopy(shape.toArray(), 0, this.shape().toArray(), 0, shape.length);
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
        this.setShape(new Shape(data.length, this.data().length / data.length));
    }

    public Matrix(Matrix other) {
        setData(new double[other.data().length]);
        System.arraycopy(other.data(), 0, data(), 0, other.data().length);

        setShape(new Shape(other.shape()));

        strides = new int[other.strides.length];
        System.arraycopy(other.strides, 0, strides, 0, other.strides.length);
    }

    protected Matrix(double[] data, Shape shape, int[] strides) {
        // used internally to create transposes, which are views
        this.setData(data);
        this.setShape(shape);
        this.strides = strides;
    }

    @Override
    protected void setShape(Shape shape) {
        // strides/indexing written generally in case tensor support is added later
        strides = new int[shape.length];
        int currStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (i != shape.length - 1) {
                currStride *= shape.axis(i + 1);
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

            if (currIdx < -shape().axis(axis) || currIdx >= shape().axis(axis)) {
                throw new ArrayIndexOutOfBoundsException(String.format(
                        "index %d out of range for axis %d of length %d", currIdx, axis, shape().axis(axis)
                ));
            } if (currIdx < 0) {
                // negative indexes are supported
                currIdx += shape().axis(axis);
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

    public boolean isSquare() {
        int refShape = shape().axis(0);
        for (int axis = 0; axis < shape().length; axis++) {
            if (refShape != shape().axis(axis)) {
                return false;
            }
        }
        return true;
    }

    protected void checkAddShapes(LinAlgObj other) throws UnsupportedOperationException {
        if (!shape().equals(other.shape())) {
            throw new UnsupportedOperationException(String.format(
                "cannot perform requested operation between shapes %s and %s", shape(), other.shape()
            ));
        }
    }

    protected void checkMulShapes(LinAlgObj other) throws UnsupportedOperationException {
        if (other.ndims() > 0 && shape().axis(ndims() - 1) != other.shape().axis(0)) {
            throw new UnsupportedOperationException(String.format(
                "cannot perform multiplication between shapes %s and %s", shape(), other.shape()
            ));
        }
    }

    private void imatMul2x2(Matrix b, Matrix newMatrix) {
        for (int i = 0; i < shape().axis(0); i++) {
            for (int j = 0; j < b.shape().axis(1); j++) {
                for (int k = 0; k < shape().axis(1); k++) {
                    int[] idxs = new int[]{i, j};
                    newMatrix.set(idxs, newMatrix.get(idxs) + get(i, k) * b.get(k, j));
                }
            }
        }
    }

    private void imatPow2x2(Scalar scalar, Matrix newMatrix) { }

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
    public <T extends LinAlgObj> Matrix mul(T other) {
        checkMulShapes(other);
        if (other.ndims() == 0) {
            // scalar multiplication is communative
            return (Matrix) (other.mul(this));
        } else {
            Matrix newMatrix = new Matrix(new double[shape().axis(0)][other.shape().axis(other.ndims() - 1)]);
            imatMul2x2((Matrix) other, newMatrix);
            return newMatrix;
        }
    }

    @Override
    public Matrix pow(Scalar scalar) {
        checkAddShapes(this);
        Matrix newMatrix = new Matrix(new double[data().length]);
        imatPow2x2(scalar, newMatrix);
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
        imatMul2x2((Matrix) other, this);
    }

    @Override
    public void ipow(Scalar scalar) {
        checkAddShapes(this);
        imatPow2x2(scalar, this);
    }

    public Matrix transpose() {
        int[] newShape = new int[ndims()];
        int[] newStrides = new int[ndims()];
        for (int axis = 0; axis < ndims(); axis++) {
            newShape[axis] = shape().axis(ndims() - axis - 1);
            newStrides[axis] = strides[ndims() - axis - 1];
        }
        return new Matrix(data(), new Shape(newShape), newStrides);
    }

    public Scalar det() {
        if (shape().equals(2, 2)) {
            return new Scalar(data()[0] * data()[3] - data()[1] * data()[2]);
        } else {
            return null;
        }
    }

    private int rowOp(int rowPos, int colPos, int pivotRowPos) {
        double factor = get(rowPos, colPos) / get(pivotRowPos, colPos);
        int numUpdated = 0;

        for (int col = 0; col < shape().axis(1); col++) {
            double updated = get(rowPos, col) - factor * get(pivotRowPos, col);
            if (col < colPos && get(rowPos, col) == 0 && updated != 0) {
                for (int rCol = 0; rCol <= numUpdated; rCol++) {
                    set(new int[]{rowPos, rCol}, 0);
                }
                return rowPos;
            } else {
                set(new int[]{rowPos, col}, updated);
                numUpdated++;
            }
        }
        return pivotRowPos;
    }

    public Matrix ref() {
        Matrix newMatrix = new Matrix(this);
        for (int col = 0; col < newMatrix.shape().axis(1); col++) {
            int pivotRowPos = -1;
            for (int row = 0; row < newMatrix.shape().axis(0); row++) {
                double currElem = newMatrix.get(row, col);
                if (currElem != 0 && pivotRowPos == -1) {
                    pivotRowPos = row;
                } else if (currElem != 0) {
                    pivotRowPos = newMatrix.rowOp(row, col, pivotRowPos);
                }
            }
        }
        return newMatrix;
    }

    public Matrix inv() throws UnsupportedOperationException {
        if (!isSquare()) {
            throw new UnsupportedOperationException("matrix must be square to be invertible");
        }
        if (shape().equals(2, 2)) {
            return det().pow(-1).mul(
                    new Matrix(new double[][]{
                        new double[]{data()[3], -data()[1]},
                        new double[]{-data()[2], data()[0]}
                    })
            );
        } else {
            return null;
        }
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (int i = 0; i < shape().axis(0); i++) {
            for (int j = 0; j < shape().axis(1); j++) {
                result.append(" ").append(String.format("%." + PRINT_PRECISION + "f", get(i, j))).append(" ");
            }
            result.append("\n");
        }
        result.setLength(result.length() - 1);
        return result.append("]").toString();
    }

}
