package com.github.orangese.linalg;

import com.github.orangese.linalg.decomp.LUDecomp;
import com.github.orangese.linalg.decomp.LUPDecomp;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Matrix extends LinAlgObj {

    public final static double EPS = 1e-6;  // only works if precision < 1e-6
    private static int PRINT_PRECISION = 3;
    private int[] strides;
    private final DecompCache decompCache;

    public Matrix(Shape shape) {
        this.setData(new double[size()]);
        this.setShapeAndStrides(shape);
        this.decompCache = new DecompCache();
    }

    public Matrix(Shape shape, double fillVal) {
        this(shape);
        Arrays.fill(data(), fillVal);
    }

    public Matrix(double... data) {
        this.setData(new double[data.length]);
        System.arraycopy(data, 0, this.data(), 0, data.length);
        this.setShapeAndStrides(new Shape(1, data.length));
        this.decompCache = new DecompCache();
    }

    public Matrix(double[] data, Shape shape) {
        this(data);
        System.arraycopy(shape.toArray(), 0, this.shape().toArray(), 0, shape.length);
        this.setShapeAndStrides(shape());
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
        this.setShapeAndStrides(new Shape(data.length, this.data().length / data.length));
        this.decompCache = new DecompCache();
    }

    public Matrix(Matrix other) {
        setData(new double[other.data().length]);
        System.arraycopy(other.data(), 0, data(), 0, other.data().length);

        setShapeAndStrides(new Shape(other.shape()));

        strides = new int[other.strides.length];
        System.arraycopy(other.strides, 0, strides, 0, other.strides.length);

        decompCache = new DecompCache(other.decompCache);
    }

    protected Matrix(double[] data, Shape shape, int[] strides, DecompCache decompCache) {
        this.setData(data);
        this.setShapeAndStrides(shape);
        this.strides = strides;
        this.decompCache = decompCache;
    }

    protected void setShapeAndStrides(Shape shape) {
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

    private int getStrided(int... idxs) {
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
        decompCache.clear();
    }

    private void setKeepCache(int[] idxs, double newVal) {
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
        if (other.ndims() > 0 && shape().axis(ndims() - 1) != other.shape().axis(0)) {
            throw new IllegalArgumentException(String.format(
                "cannot perform %s between shapes %s and %s", op, shape(), other.shape()
            ));
        }
    }

    private void imatMul2Axis(Matrix mat, Matrix newMatrix) {
        for (int i = 0; i < shape().axis(0); i++) {
            for (int j = 0; j < mat.shape().axis(1); j++) {
                for (int k = 0; k < shape().axis(1); k++) {
                    newMatrix.setKeepCache(new int[]{i, j}, newMatrix.get(i, j) + get(i, k) * mat.get(k, j));
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
            Matrix newMatrix = new Matrix(new double[shape().axis(0)][other.shape().axis(other.ndims() - 1)]);
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
        int[] newShape = new int[ndims()];
        int[] newStrides = new int[ndims()];
        for (int axis = 0; axis < ndims(); axis++) {
            newShape[axis] = shape().axis(ndims() - axis - 1);
            newStrides[axis] = strides[ndims() - axis - 1];
        }
        return new Matrix(data(), new Shape(newShape), newStrides, decompCache);
    }

    public Scalar det() {
        if (shape().equals(2, 2)) {
            return new Scalar(data()[0] * data()[3] - data()[1] * data()[2]);
        } else {
            return null;
        }
    }

    private void rowOp(int rowPos, int colPos, int pivotRowPos, Matrix elimMatrixInv) {
        final double factor = get(rowPos, colPos) / get(pivotRowPos, colPos);

        int[] elimPos = new int[]{rowPos, pivotRowPos};
        double elimRollBack = 0;
        if (elimMatrixInv != null) {
            elimRollBack = elimMatrixInv.get(elimPos);
            elimMatrixInv.set(elimPos, elimRollBack + factor);
        }

        int numUpdated = 0;
        for (int col = 0; col < shape().axis(1); col++) {
            double updated = get(rowPos, col) - factor * get(pivotRowPos, col);

            if (col < colPos && get(rowPos, col) == 0 && updated != 0) {
                // roll back changes if row op isn't valid
                if (elimMatrixInv != null) {
                    elimMatrixInv.set(elimPos, elimRollBack);
                } for (int rCol = 0; rCol <= numUpdated; rCol++) {
                    setKeepCache(new int[]{rowPos, rCol}, 0);
                }
                break;
            } else {
                setKeepCache(new int[]{rowPos, col}, updated);
                numUpdated++;
            }
        }
    }

    private void swapRows(int rowA, int rowB) {
        double[] tmp = new double[shape().axis(1)];
        System.arraycopy(data(), getStrided(rowA, 0), tmp, 0, tmp.length);

        System.arraycopy(data(), getStrided(rowB, 0), data(), getStrided(rowA, 0), tmp.length);
        System.arraycopy(tmp, 0, data(), getStrided(rowB, 0), tmp.length);
    }

    private LUPDecomp LUPDecompBackend(String mode) {
        Matrix ref = new Matrix(this);

        Matrix permMatrix = mode.contains("lu") ? Matrix.eye(shape()) : null;
        Matrix elimMatrixInv = mode.contains("lu") ? Matrix.eye(shape()) : null;

        for (int col = 0; col < shape().axis(1); col++) {
            int pivotRow = -1;
            int zeroRow = -1;

            for (int row = col; row < shape().axis(0); row++) {
                double currElem = ref.get(row, col);

                if (Scalar.isNonZero(currElem)) {
                    if (pivotRow == -1) {
                        pivotRow = row;
                    } else {
                        ref.rowOp(row, col, pivotRow, elimMatrixInv);
                        zeroRow = row;
                    } if (zeroRow != -1) {
                        if (mode.equals("lu")) {
                            throw new UnsupportedOperationException("permutation not allowed in LU decomposition");
                        } if (pivotRow == row) {
                            pivotRow = zeroRow;
                        } if (permMatrix != null) {
                            permMatrix.swapRows(zeroRow, row);
                        }
                        ref.swapRows(zeroRow, row);
                        zeroRow = row;
                    }
                } if (zeroRow == -1) {
                    zeroRow = row;
                }

                decompCache.addPivot(pivotRow);
                if (elimMatrixInv != null) {
                    elimMatrixInv.decompCache.addPivot(pivotRow);
                }
            }
        }
        return new LUPDecomp(permMatrix, elimMatrixInv, ref);
    }

    public LUDecomp LUDecomp() {
        if (!decompCache.computedLUP()) {
            decompCache.setLUPDecomp(LUPDecompBackend("lu"));
        }
        return decompCache.getLUDecompCopy();
    }

    public LUPDecomp LUPDecomp() {
        if (!decompCache.computedLUP()) {
            decompCache.setLUPDecomp(LUPDecompBackend("lup"));
        }
        return (LUPDecomp) (decompCache.getLUDecompCopy());
    }

    public Matrix ref() {
        if (!decompCache.computedLUP()) {
            decompCache.setLUPDecomp(LUPDecompBackend("ref"));
        }
        return decompCache.getLUDecompView().U();
    }

    public int[] pivots() {
        if (!decompCache.computedPivots()) {
            decompCache.setLUPDecomp(LUPDecompBackend("lup"));
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
        if (shape.length != 2) {
            throw new IllegalArgumentException("cannot instantiate identity matrix with number of axis != 2");
        }

        double[] data = new double[shape.size()];
        int dataOffset = 0;

        for (int i = 0; i < data.length; i++) {
            if (i == dataOffset + offset) {
                data[i] = 1;
                dataOffset += shape.axis(1) + 1;
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
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } if (!(o instanceof Matrix)) {
            return false;
        } if (!super.equals(o)) {
            return false;
        }
        return Arrays.equals(strides, ((Matrix) o).strides);
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

class DecompCache {

    private Set<Integer> pivots;
    private LUDecomp luDecomp;

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

    public void setLUPDecomp(LUDecomp luDecomp) {
        this.luDecomp = luDecomp;
    }

    public LUDecomp getLUDecompView() {
        return luDecomp;
    }

    public LUDecomp getLUDecompCopy() {
        return luDecomp.deepcopy();
    }

    public boolean computedLUP() {
        return luDecomp != null;
    }

    public boolean computedPivots() {
        return !pivots.isEmpty();
    }
}
