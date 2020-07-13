package com.github.orangese.linalg;

import com.github.orangese.linalg.decomp.LUPDecomp;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Matrix extends LinAlgObj {

    public final static double EPS = 1e-10;  // only works if precision < 1e-6
    private static int PRINT_PRECISION = 3;
    private final DecompCache decompCache;

    public Matrix(Shape shape) {
        this.setData(new double[shape.size()]);
        this.setShape(shape);
        this.decompCache = new DecompCache();
    }

    public Matrix(Shape shape, double fillVal) {
        this(shape);
        Arrays.fill(data(), fillVal);
    }

    public Matrix(double... data) {
        this.setData(new double[data.length]);
        System.arraycopy(data, 0, this.data(), 0, data.length);
        this.setShape(new Shape(1, data.length));
        this.decompCache = new DecompCache();
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
        this.decompCache = new DecompCache();
    }

    public Matrix(Matrix other) {
        setData(new double[other.data().length]);
        System.arraycopy(other.data(), 0, data(), 0, other.data().length);
        setShape(new Shape(other.shape()));
        decompCache = new DecompCache(other.decompCache);
    }

    protected Matrix(double[] data, Shape shape, boolean view) {
        this.setData(data);
        this.setShape(shape);
        this.decompCache = null;  // decomp(A) != decomp(A^T)
    }

    private int getStrided(int row, int col) {
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
        setKeepCache(row, col, newVal);
        decompCache.clear();
    }

    private void setKeepCache(int row, int col, double newVal) {
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
                for (int k = 0; k < colDim(); k++) {
                    newMatrix.setKeepCache(i, j, newMatrix.get(i, j) + get(i, k) * mat.get(k, j));
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
        imatMul2Axis((Matrix) other, this);
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
        if (shape().equals(2, 2)) {
            return new Scalar(data()[0] * data()[3] - data()[1] * data()[2]);
        } else {
            return null;
        }
    }

    private double backwardSolveLUP(int i, int j) {
        double sum = get(i, j);
        for (int k = 0; k < Math.min(i, j); k++) {
            sum -= get(i, k) * get(k, j);
        }
        setKeepCache(i, j, sum);
        return sum;
    }

    private LUPDecomp decompLUP(String mode) {
        if (!isSquare()) {
            throw new UnsupportedOperationException("nonsquare LU decomp not yet implemented");
        }

        Matrix decomp = new Matrix(this);

        int[] permutations = null;
        if (!mode.equals("lu")) {
            permutations = new int[rowDim()];
            for (int i = 0; i < permutations.length; i++) {
                permutations[i] = i;
            }
        }

        for (int j = 0; j < colDim(); j++) {
            for (int i = 0; i < j; i++) {
                decomp.backwardSolveLUP(i, j);
            }

            int max = j;
            double largest = Double.NEGATIVE_INFINITY;
            for (int i = j; i < rowDim(); i++) {
                double sum = decomp.backwardSolveLUP(i, j);
                if (mode.contains("lu") && Math.abs(sum) > largest) {
                    largest = Math.abs(sum);
                    max = i;
                }
            }

            if (Math.abs(decomp.get(max, j)) < EPS) {
                throw new ArithmeticException("matrix is singular");
            }

            if (permutations != null && max != j) {
                double[] tmpArr = new double[colDim()];
                int len = tmpArr.length;

                System.arraycopy(decomp.data(), getStrided(max, 0), tmpArr, 0, len);
                System.arraycopy(decomp.data(), getStrided(j, 0), decomp.data(), getStrided(max, 0), len);
                System.arraycopy(tmpArr, 0, decomp.data(), getStrided(j, 0), len);

                int tmp = permutations[max];
                permutations[max] = permutations[j];
                permutations[j] = tmp;
            }

            double diag = decomp.get(j, j);
            for (int i = j + 1; i < rowDim(); i++) {
                decomp.setKeepCache(i, j, decomp.get(i, j) / diag);
            }

        }
        return new LUPDecomp(decomp, permutations);
    }

    public LUPDecomp decompLUP() {
        if (!decompCache.computedLUP()) {
            decompCache.setLUPDecomp(decompLUP("lup"));
        }
        return decompCache.getLUDecompCopy();
    }

    public Matrix ref() {
        if (!decompCache.computedLUP()) {
            decompCache.setLUPDecomp(decompLUP("ref"));
        }
        return decompCache.getLUDecompView().U();
    }

    public int[] pivots() {
        if (!decompCache.computedPivots()) {
            decompCache.setLUPDecomp(decompLUP("lup"));
        }
        return decompCache.getPivots().stream().mapToInt(x -> x).toArray();
    }

    public Matrix inv() {
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

class DecompCache {

    private Set<Integer> pivots;
    private LUPDecomp luDecomp;

    public DecompCache() {
        clear();
    }

    public DecompCache(DecompCache other) {
        pivots = other.pivots == null ? null : new HashSet<>(other.pivots);
        // lupDecomp will only be edited via re-assignment to a new Matrix reference, so not copying is okay
        luDecomp = other.luDecomp;
    }

    public void clear() {
        pivots = null;
        luDecomp = null;
    }

    public void addPivot(int pivot) {
        if (pivots == null) {
            pivots = new HashSet<>();
        } if (pivot != -1) {
            pivots.add(pivot);
        }
    }

    public Set<Integer> getPivots() {
        return pivots;
    }

    public void setLUPDecomp(LUPDecomp luDecomp) {
        this.luDecomp = luDecomp;
    }

    public LUPDecomp getLUDecompView() {
        return luDecomp;
    }

    public LUPDecomp getLUDecompCopy() {
        return luDecomp.deepcopy();
    }

    public boolean computedLUP() {
        return luDecomp != null;
    }

    public boolean computedPivots() {
        return pivots != null && !pivots.isEmpty();
    }
}
