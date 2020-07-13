package com.github.orangese.linalg;

public class LUPDecomp {

    public final static double EPS = 1e-10;
    private final Matrix decomp;
    private final int[] permArray;
    private Matrix lower;
    private Matrix upper;
    private Matrix perm;

    public LUPDecomp(Matrix mat) {
        this(mat, false);
    }

    protected LUPDecomp(Matrix mat, boolean ref) {
        if (!mat.isSquare()) {
            throw new UnsupportedOperationException("nonsquare LU decomp not yet implemented");
        }

        decomp = new Matrix(mat);

        permArray = new int[mat.rowDim()];
        if (!ref) {
            for (int i = 0; i < permArray.length; i++) {
                permArray[i] = i;
            }
        }

        for (int j = 0; j < mat.colDim(); j++) {
            for (int i = 0; i < j; i++) {
                backwardSolve(i, j, decomp);
            }

            int max = j;
            if (!ref) {
                double largest = Double.NEGATIVE_INFINITY;
                for (int i = j; i < decomp.rowDim(); i++) {
                    double sum = backwardSolve(i, j, decomp);
                    if (Math.abs(sum) > largest) {
                        largest = Math.abs(sum);
                        max = i;
                    }
                }
            }

            if (Math.abs(decomp.get(max, j)) < EPS) {
                throw new ArithmeticException("matrix is singular");
            }

            if (!ref && max != j) {
                double[] tmpArr = new double[decomp.colDim()];

                System.arraycopy(decomp.data(), decomp.getStrided(max, 0), tmpArr, 0, tmpArr.length);
                System.arraycopy(decomp.data(), decomp.getStrided(j, 0), decomp.data(),
                        decomp.getStrided(max, 0), tmpArr.length);
                System.arraycopy(tmpArr, 0, decomp.data(), decomp.getStrided(j, 0), tmpArr.length);

                int tmp = permArray[max];
                permArray[max] = permArray[j];
                permArray[j] = tmp;
            }

            double diag = decomp.get(j, j);
            for (int i = j + 1; i < decomp.rowDim(); i++) {
                decomp.set(i, j, decomp.get(i, j) / diag);
            }
        }
    }

    private double backwardSolve(int i, int j, Matrix mat) {
        double sum = mat.get(i, j);
        for (int k = 0; k < Math.min(i, j); k++) {
            sum -= mat.get(i, k) * mat.get(k, j);
        }
        mat.set(i, j, sum);
        return sum;
    }

    public Matrix L() {
        if (lower == null) {
            lower = new Matrix(decomp.shape());
            for (int j = 0; j < lower.colDim(); j++) {
                for (int i = j; i < lower.rowDim(); i++) {
                    lower.set(i, j, decomp.get(i, j));
                }
                lower.set(j, j, 1);
            }
        }
        return lower;
    }

    public Matrix U() {
        if (upper == null) {
            upper = new Matrix(decomp.shape());
            for (int j = 0; j < lower.colDim(); j++) {
                for (int i = 0; i < j + 1; i++) {
                    upper.set(i, j, decomp.get(i, j));
                }
            }
        }
        return upper;
    }

    public Matrix P() {
        if (perm == null) {
            perm = new Matrix(new Shape(permArray.length, permArray.length));
            for (int i = 0; i < permArray.length; i++) {
                perm.set(i, permArray[i], 1);
            }
        }
        return perm;
    }

    @Override
    public String toString() {
        return super.toString() + "\nP: " + P();
    }

}
