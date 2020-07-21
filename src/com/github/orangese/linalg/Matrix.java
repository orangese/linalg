package com.github.orangese.linalg;

import java.util.Arrays;

public class Matrix extends LinAlgObj {

    private static int PRINT_PRECISION = 3;
    private int[] strides;

    public Matrix(Shape shape) {
        setData(new double[shape.size()]);
        setShape(shape);
        calcStrides();
    }

    public Matrix(Shape shape, double fillVal) {
        this(shape);
        Arrays.fill(data(), fillVal);
    }

    public Matrix(double... data) {
        setData(new double[data.length]);
        System.arraycopy(data, 0, data(), 0, data.length);
        setShape(new Shape(1, data.length));
        calcStrides();
    }

    public Matrix(double[] data, Shape shape) {
        this(data);
        setShape(shape);
        calcStrides();
    }

    public Matrix(double[][] data) {
        setData(new double[data.length * data[0].length]);
        for (int axis = 0; axis < data.length; axis++) {
            if (data[axis].length != data[0].length) {
                throw new IllegalArgumentException(String.format(
                        "dim %d along axis %d != dim %d along axis 0", data[axis].length, axis, data[0].length
                ));
            }
            System.arraycopy(data[axis], 0, data(), data[axis].length * axis, data[axis].length);
        }
        setShape(new Shape(data.length, data[0].length));
        calcStrides();
    }

    public Matrix(Vector... vectors) {
        this(Arrays.stream(vectors).map(LinAlgObj::data).toArray(double[][]::new));
    }

    public Matrix(Matrix o) {
        setData(new double[o.data().length]);
        System.arraycopy(o.data(), 0, data(), 0, o.data().length);
        setShape(new Shape(o.shape()));
        strides = o.strides;
    }

    private void calcStrides() {
        strides = new int[]{colDim(), 1};
    }

    protected int getStrided(int row, int col) {
        if (row < -rowDim() || row >= rowDim()) {
            throw new ArrayIndexOutOfBoundsException(String.format(
                    "index %d out of range for axis %d of length %d", row, 0, rowDim()
            ));
        } else if (col < -colDim() || col >= colDim()) {
            throw new ArrayIndexOutOfBoundsException(String.format(
                    "index %d out of range for axis %d of length %d", col, 1, colDim()
            ));
        }

        // negative indexes are supported
        if (row < 0) {
            row += rowDim();
        } if (col < 0) {
            col += colDim();
        }
        return row * strides[0] + col * strides[1];
    }

    public double get(int row, int col) {
        return data()[getStrided(row, col)];
    }

    public Vector getRow(int row) {
        Vector rowVec = new Vector(new Shape(1, colDim()));
        System.arraycopy(data(), getStrided(row, 0), rowVec.data(), 0, rowVec.size());
        return rowVec;
    }

    public Vector getCol(int col) {
        Vector colVec = new Vector(new Shape(rowDim(), 1));
        for (int i = 0; i < rowDim(); i++) {
            colVec.set(i, get(i, col));
        }
        return colVec;
    }

    public void set(int row, int col, double newVal) {
        data()[getStrided(row, col)] = newVal;
    }

    public boolean isNotSquare() {
        return rowDim() != colDim();
    }

    @Override
    protected void checkAddShapes(LinAlgObj o, String op) {
        if (!shape().equals(o.shape())) {
            throw new IllegalArgumentException(String.format(
                "cannot perform %s between shapes %s and %s", op, shape(), o.shape()
            ));
        }
    }

    @Override
    protected void checkMulShapes(LinAlgObj o, String op) {
        if (o.ndims() > 0 && colDim() != o.rowDim()) {
            throw new IllegalArgumentException(String.format(
                "cannot perform %s between shapes %s and %s", op, shape(), o.shape()
            ));
        }
    }

    private void imatMul2Axis(Matrix mat, Matrix newMatrix) {
        for (int i = 0; i < rowDim(); i++) {
            for (int j = 0; j < mat.colDim(); j++) {
                for (int k = 0; k <  colDim(); k++) {
                    newMatrix.set(i, j, newMatrix.get(i, j) + get(i, k) * mat.get(k, j));
                }
            }
        }
    }

    private void imatPow2Axis(Scalar scalar, Matrix newMatrix) { }

    @Override
    public Matrix add(LinAlgObj o) {
        checkAddShapes(o, "matrix addition");
        Matrix newMat = new Matrix(shape());
        for (int i = 0; i < data().length; i++) {
            newMat.data()[i] = data()[i] + o.data()[i];
        }
        return newMat;
    }

    @Override
    public Matrix subtract(LinAlgObj o) {
        checkAddShapes(o, "matrix subtraction");
        Matrix newMat = new Matrix(shape());
        for (int i = 0; i < data().length; i++) {
            newMat.data()[i] = data()[i] - o.data()[i];
        }
        return newMat;
    }

    @Override
    public Matrix mul(Matrix o) {
        checkMulShapes(o, "matrix multiplication");
        if (o.ndims() == 0) {
            // scalar multiplication is communative
            return o.mul(this);
        } else {
            Matrix newMatrix = new Matrix(new double[rowDim()][o.colDim()]);
            imatMul2Axis(o, newMatrix);
            return newMatrix;
        }
    }

    @Override
    public Matrix pow(Scalar scalar) {
        checkAddShapes(this, "matrix power");
        Matrix newMatrix = new Matrix(new double[data().length]);
        imatPow2Axis(scalar, newMatrix);
        return newMatrix;
    }

    @Override
    public void iadd(LinAlgObj o) {
        checkAddShapes(o, "matrix addition");
        for (int i = 0; i < data().length; i++) {
            data()[i] += o.data()[i];
        }
    }

    @Override
    public void isubtract(LinAlgObj o) {
        checkAddShapes(o, "matrix subtraction");
        for (int i = 0; i < data().length; i++) {
            data()[i] -= o.data()[i];
        }
    }

    @Override
    public void imul(LinAlgObj o) {
        checkAddShapes(o, "matrix multiplication");
        Matrix tmp = viewOf(this);
        setData(new double[data().length]);
        tmp.imatMul2Axis((Matrix) o, this);
    }

    @Override
    public void ipow(Scalar scalar) {
        checkAddShapes(this, "matrix power");
        imatPow2Axis(scalar, this);
    }

    public Matrix transpose() {
        Matrix transpose = viewOf(this);

        transpose.setShape(new Shape(colDim(), rowDim()));
        transpose.strides = new int[]{1, colDim()};

        return transpose;
    }

    public Scalar trace() {
        if (isNotSquare()) {
            throw new UnsupportedOperationException("cannot compute trace for nonsquare matrix");
        }
        double trace = 1;
        for (int i = 0; i < rowDim(); i++) {
            trace *= get(i, i);
        }
        return new Scalar(trace);
    }

    public Scalar item() {
        if (!shape().equals(1, 1)) {
            throw new IllegalArgumentException("cannot instantiate Scalar from LinAlgObj with shape " + shape());
        }
        return new Scalar(get(0, 0));
    }

    public static int getPrintPrecision() {
        return PRINT_PRECISION;
    }

    public static void setPrintPrecision(int printPrecision) {
        if (printPrecision <= 0) {
            throw new IllegalArgumentException("printPrecision must be an int greater than 0");
        }
        PRINT_PRECISION = printPrecision;
    }

    public static Matrix eye(Shape shape, int offset) {
        if (shape.ndims() != 2) {
            throw new IllegalArgumentException("cannot instantiate identity matrix with number of axis != 2");
        }

        double[] data = new double[shape.size()];
        int dataOffset = 0;

        for (int i = 0; i < data.length; i++) {
            if (i == dataOffset + offset) {
                data[i] = 1;
                dataOffset += shape.colDim() + 1;
            }
        }

        return new Matrix(data, shape);
    }

    public static Matrix eye(Shape shape) {
        return eye(shape, 0);
    }

    public static Matrix viewOf(Matrix o) {
        Matrix view = new Matrix(o.shape());

        view.setData(o.data());
        view.setShape(o.shape());
        view.strides = o.strides;

        return view;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (int i = 0; i < rowDim(); i++) {
            for (int j = 0; j < colDim(); j++) {
                result.append(" ").append(String.format("%." + PRINT_PRECISION + "f", get(i, j))).append(" ");
            }
            result.append("\n ");
        }
        result.setLength(result.length() - 2);
        return result.append("]").toString();
    }

}
