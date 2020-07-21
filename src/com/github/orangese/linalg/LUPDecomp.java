package com.github.orangese.linalg;

import java.util.ArrayList;
import java.util.List;

public class LUPDecomp {

    public final static double EPS = 1e-10;
    private final Matrix decomp;
    private final int[] permArray;
    private final List<Integer> pivotPos;
    private Matrix lower;
    private Matrix upper;
    private Matrix perm;
    private int numPermutations;
    private boolean singular;

    public LUPDecomp(Matrix mat) {
        decomp = new Matrix(mat);

        pivotPos = new ArrayList<>();
        singular = mat.isNotSquare();
        numPermutations = 0;

        permArray = new int[mat.rowDim()];
        for (int i = 0; i < permArray.length; i++) {
            permArray[i] = i;
        }

        for (int j = 0; j < mat.colDim(); j++) {
            for (int i = 0; i < j; i++) {
                backwardSolve(i, j);
            }

            int max = j;
            double largest = Double.NEGATIVE_INFINITY;
            for (int i = j; i < decomp.rowDim(); i++) {
                double sum = backwardSolve(i, j);
                if (Math.abs(sum) > largest) {
                    largest = Math.abs(sum);
                    max = i;
                }
            }

            boolean isZero = j < mat.rowDim() && Math.abs(decomp.get(max, j)) < EPS;
            if (max != j || isZero) {
                if (isZero) {
                    singular = true;
                }

                double[] tmpArr = new double[decomp.colDim()];

                System.arraycopy(decomp.data(), decomp.getStrided(max, 0), tmpArr, 0, tmpArr.length);
                System.arraycopy(decomp.data(), decomp.getStrided(j, 0), decomp.data(),
                        decomp.getStrided(max, 0), tmpArr.length);
                System.arraycopy(tmpArr, 0, decomp.data(), decomp.getStrided(j, 0), tmpArr.length);

                int tmp = permArray[max];
                permArray[max] = permArray[j];
                permArray[j] = tmp;

                numPermutations++;
            }

            if (j < mat.rowDim() && j < mat.colDim()) {
                final double diag = decomp.get(j, j);
                if (diag != 0) {
                    pivotPos.add(j);
                    for (int i = j + 1; i < decomp.rowDim(); i++) {
                        decomp.set(i, j, decomp.get(i, j) / diag);
                    }
                }
            }
        }
    }

    private double backwardSolve(int i, int j) {
        try {
            double sum = decomp.get(i, j);
            for (int k = 0; k < Math.min(i, j); k++) {
                sum -= decomp.get(i, k) * decomp.get(k, j);
            }
            decomp.set(i, j, sum);
            return sum;
        } catch (ArrayIndexOutOfBoundsException ignored) {
            return 0;
        }
    }

    public Matrix L() {
        if (lower == null) {
            lower = new Matrix(new Shape(decomp.rowDim(), decomp.rowDim()));
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
            for (int j = 0; j < upper.colDim(); j++) {
                for (int i = 0; i < Math.min(j + 1, upper.rowDim()); i++) {
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

    public Matrix solve(Matrix b) {
        if (singular) {
            throw new UnsupportedOperationException("matrix is singular");
        } else if (decomp.rowDim() != b.rowDim()) {
            throw new IllegalArgumentException("equation is not solveable for LHS with shape " + decomp.shape() +
                    " and RHS with shape " + b.shape());
        }

        Matrix x = new Matrix(new Shape(decomp.colDim(), b.colDim()));

        for (int i = 0; i < decomp.rowDim(); i++) {
            for (int j = 0; j < b.colDim(); j++) {
                x.set(i, j, b.get(permArray[i], j));
            }
        }

        for (int j = 0; j < decomp.rowDim(); j++) {
            for (int i = j + 1; i < decomp.rowDim(); i++) {
                final double factor = decomp.get(i, j);
                for (int k = 0; k < b.colDim(); k++) {
                    x.set(i, k, x.get(i, k) - x.get(j, k) * factor);
                }
            }
        }

        for (int j = decomp.rowDim() - 1; j >= 0; j--) {
            final double diag = decomp.get(j, j);
            for (int k = 0; k < b.colDim(); k++) {
                x.set(j, k, x.get(j, k) / diag);
            }
            for (int i = 0; i < j; i++) {
                final double factor = decomp.get(i, j);
                for (int k = 0; k < b.colDim(); k++) {
                    x.set(i, k, x.get(i, k) - x.get(j, k) * factor);
                }
            }
        }

        return x;
    }

    public Vector solve(Vector b) {
        return Vector.asVector(solve(Matrix.viewOf(b)));
    }

    public Matrix rref() {
        final Matrix ref = U();

        int prevPivotPos = -1;
        for (int j = 0; j < decomp.colDim(); j++) {
            boolean foundPivot = false;

            for (int i = Math.min(j, decomp.rowDim() - 1); i >= 0; i--) {
                final double currElem = ref.get(i, j);

                if (Math.abs(currElem) >= LUPDecomp.EPS) {
                    if (!foundPivot && i <= prevPivotPos) {
                        break;

                    } else if (!foundPivot) {
                        for (int k = 0; k < decomp.colDim(); k++) {
                            ref.set(i, k, ref.get(i, k) / currElem);
                        }
                        prevPivotPos = i;
                        foundPivot = true;

                    } else {
                        final double factor = ref.get(i, j);
                        for (int k = j; k < decomp.colDim(); k++) {
                            ref.set(i, k, ref.get(i, k) - factor * ref.get(prevPivotPos, k));
                        }
                    }
                }
            }
        }

        return ref;
    }

    public int rank() {
        return pivotPos.size();
    }

    public List<Integer> getPivotPos() {
        return pivotPos;
    }

    public Matrix inv() {
        return solve(Matrix.eye(decomp.shape()));
    }

    public Scalar det() {
        if (decomp.isNotSquare()) {
            throw new UnsupportedOperationException("cannot compute determinant for nonsquare matrix");
        }
        if (isSingular()) {
            return new Scalar(0);
        } else {
            double coef = Math.pow(-1, numPermutations);
            return U().trace().mul(coef);
        }
    }

    public boolean isSingular() {
        return singular;
    }

    @Override
    public String toString() {
        return "L: " + L() + "\nU: " + U() + "\nP: " + P();
    }

}
