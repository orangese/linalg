package com.github.orangese.linalg;

public class LUPDecomp {

    public final static double EPS = 1e-10;
    private final Matrix decomp;
    private final int[] permArray;
    private Matrix lower;
    private Matrix upper;
    private Matrix perm;
    private int numPermutations;

    public LUPDecomp(Matrix mat) {
        decomp = new Matrix(mat);

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

            if (max != j || j < mat.rowDim() && Math.abs(decomp.get(max, j)) < EPS) {
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
                double diag = decomp.get(j, j);
                for (int i = j + 1; i < decomp.rowDim(); i++) {
                    decomp.set(i, j, decomp.get(i, j) / diag);
                }
            }
        }
    }

    private double backwardSolve(int i, int j) {
        double sum = decomp.get(i, j);
        for (int k = 0; k < Math.min(i, j); k++) {
            sum -= decomp.get(i, k) * decomp.get(k, j);
        }
        decomp.set(i, j, sum);
        return sum;
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

    protected int getNumPermutations() {
        return numPermutations;
    }

    @Override
    public String toString() {
        return "L: " + L() + "\nU: " + U() + "\nP: " + P();
    }

}
