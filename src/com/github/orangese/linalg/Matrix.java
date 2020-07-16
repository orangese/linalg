package com.github.orangese.linalg;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class Matrix extends LinAlgObj {

    private static int PRINT_PRECISION = 3;

    public Matrix(Shape shape) {
        this.setData(new double[shape.size()]);
        this.setShape(shape);
    }

    public Matrix(Shape shape, double fillVal) {
        this(shape);
        Arrays.fill(data(), fillVal);
    }

    public Matrix(double... data) {
        this.setData(new double[data.length]);
        System.arraycopy(data, 0, this.data(), 0, data.length);
        this.setShape(new Shape(1, data.length));
    }

    public Matrix(double[] data, Shape shape) {
        this(data);
        this.setShape(shape);
    }

    public Matrix(double[][] data) {
        this.setData(new double[data.length * data[0].length]);
        for (int axis = 0; axis < data.length; axis++) {
            if (data[axis].length != data[0].length) {
                throw new IllegalArgumentException(String.format(
                        "dim %d along axis %d != dim %d along axis 0", data[axis].length, axis, data[0].length
                ));
            }
            System.arraycopy(data[axis], 0, this.data(), data[axis].length * axis, data[axis].length);
        }
        this.setShape(new Shape(data.length, data[0].length));
    }

    public Matrix(Matrix other) {
        setData(new double[other.data().length]);
        System.arraycopy(other.data(), 0, data(), 0, other.data().length);
        setShape(new Shape(other.shape()));
    }

    protected Matrix(double[] data, Shape shape, boolean view) {
        this.setData(data);
        this.setShape(shape);
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
        return row * colDim() + col;
    }

    public double get(int row, int col) {
        return data()[getStrided(row, col)];
    }

    public void set(int row, int col, double newVal) {
        data()[getStrided(row, col)] = newVal;
    }

    public boolean isSquare() {
        return rowDim() == colDim();
    }

    @Override
    protected void checkAddShapes(LinAlgObj other, String op) {
        if (!shape().equals(other.shape())) {
            throw new IllegalArgumentException(String.format(
                "cannot perform %s between shapes %s and %s", op, shape(), other.shape()
            ));
        }
    }

    @Override
    protected void checkMulShapes(LinAlgObj other, String op) {
        if (other.ndims() > 0 && colDim() != other.rowDim()) {
            throw new IllegalArgumentException(String.format(
                "cannot perform %s between shapes %s and %s", op, shape(), other.shape()
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
    public <T extends LinAlgObj> Matrix add(T other) {
        checkAddShapes(other, "matrix addition");
        double[] newData = new double[data().length];
        for (int i = 0; i < data().length; i++) {
            newData[i] = data()[i] + other.data()[i];
        }
        return new Matrix(newData, shape());
    }

    @Override
    public <T extends LinAlgObj> Matrix subtract(T other) {
        checkAddShapes(other, "matrix subtraction");
        double[] newData = new double[data().length];
        for (int i = 0; i < data().length; i++) {
            newData[i] = data()[i] - other.data()[i];
        }
        return new Matrix(newData, shape());
    }

    @Override
    public <T extends LinAlgObj> Matrix mul(T other) {
        checkMulShapes(other, "matrix multiplication");
        if (other.ndims() == 0) {
            // scalar multiplication is communative
            return (Matrix) (other.mul(this));
        } else {
            Matrix newMatrix = new Matrix(new double[rowDim()][other.colDim()]);
            imatMul2Axis((Matrix) other, newMatrix);
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
    public <T extends LinAlgObj> void iadd(T other) {
        checkAddShapes(other, "matrix addition");
        for (int i = 0; i < data().length; i++) {
            data()[i] += other.data()[i];
        }
    }

    @Override
    public <T extends LinAlgObj> void isubtract(T other) {
        checkAddShapes(other, "matrix subtraction");
        for (int i = 0; i < data().length; i++) {
            data()[i] -= other.data()[i];
        }
    }

    @Override
    public <T extends LinAlgObj> void imul(T other) {
        checkAddShapes(other, "matrix multiplication");
        Matrix tmp = new Matrix(data(), shape(), true);
        setData(new double[data().length]);
        tmp.imatMul2Axis((Matrix) other, this);
    }

    @Override
    public void ipow(Scalar scalar) {
        checkAddShapes(this, "matrix power");
        imatPow2Axis(scalar, this);
    }

    public Matrix transpose() {
        return new Matrix(data(), new Shape(colDim(), rowDim()), true);
    }

    public Scalar det() {
        if (!isSquare()) {
            throw new UnsupportedOperationException("cannot compute determinant for nonsquare matrix");
        }
        if (shape().equals(2, 2)) {
            return new Scalar(data()[0] * data()[3] - data()[1] * data()[2]);
        } else {
            LUPDecomp lupDecomp = new LUPDecomp(this);
            double coef = Math.pow(-1, lupDecomp.getNumPermutations());
            return lupDecomp.U().trace().mul(coef);
        }
    }

    public Matrix ref() {
        return new LUPDecomp(this).U();
    }

    private Matrix rref(List<Integer> pivotCache, LUPDecomp computedDecomp) {
        final Matrix ref = computedDecomp != null ? computedDecomp.U() : ref();

        Matrix inv = null;
        if (computedDecomp != null) {
            computedDecomp.L().imul(computedDecomp.P());
            System.out.println("EEEEE" + computedDecomp.L());
            inv = computedDecomp.L();
        }

        int prevPivotPos = -1;
        for (int j = 0; j < colDim(); j++) {
            boolean foundPivot = false;

            for (int i = Math.min(j, rowDim() - 1); i >= 0; i--) {
                final double currElem = ref.get(i, j);

                if (Math.abs(currElem) >= LUPDecomp.EPS) {
                    if (!foundPivot && i <= prevPivotPos) {
                        break;

                    } else if (!foundPivot) {
                        for (int k = 0; k < colDim(); k++) {
                            ref.set(i, k, ref.get(i, k) / currElem);
                            if (inv != null) {
                                inv.set(i, k, inv.get(i, k) / currElem);
                            }
                        }
                        prevPivotPos = i;
                        foundPivot = true;

                        if (pivotCache != null) {
                            pivotCache.add(i);
                        }

                    } else {
                        final double factor = ref.get(i, j);
                        for (int k = j; k < colDim(); k++) {
                            ref.set(i, k, ref.get(i, k) - factor * ref.get(prevPivotPos, k));
                            if (inv != null) {
                                inv.set(i, k, inv.get(i, k) - factor * inv.get(prevPivotPos, k));
                            }
                        }
                    }
                }
            }
        }

        return ref;
    }

    public Matrix rref() {
        return rref( null, null);
    }

    public int rank() {
        return pivotPos().size();
    }

    public List<Integer> pivotPos() {
        List<Integer> pivots = new ArrayList<>();
        rref(pivots, null);
        return pivots;
    }

    public Matrix inv() {
        if (!isSquare()) {
            throw new UnsupportedOperationException("cannot compute inverse of nonsquare matrix");
        }

        if (shape().equals(2, 2)) {
            return det().pow(-1).mul(
                    new Matrix(new double[][]{
                        new double[]{data()[3], -data()[1]},
                        new double[]{-data()[2], data()[0]}
                    })
            );
        } else {
            LUPDecomp lupDecomp = new LUPDecomp(this);
            rref(null, lupDecomp);
            return lupDecomp.L();
        }
    }

    public Scalar trace() {
        if (!isSquare()) {
            throw new UnsupportedOperationException("cannot compute trace for nonsquare matrix");
        }
        double trace = 1;
        for (int i = 0; i < rowDim(); i++) {
            trace *= get(i, i);
        }
        return new Scalar(trace);
    }

    public Scalar item() {
        if (ndims() != 0 && !shape().equals(1, 1)) {
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

    public static Matrix eye(int n, int m, int offset) {
        return eye(new Shape(n, m), offset);
    }

    public static Matrix eye(int n, int m) {
        return eye(new Shape(n, m), 0);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (int i = 0; i < rowDim(); i++) {
            for (int j = 0; j < colDim(); j++) {
                result.append(" ").append(String.format("%." + PRINT_PRECISION + "f", get(i, j))).append(" ");
            }
            result.append("\n");
        }
        result.setLength(result.length() - 1);
        return result.append("]").toString();
    }

}
