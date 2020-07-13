package com.github.orangese.linalg.decomp;

import com.github.orangese.linalg.Matrix;
import com.github.orangese.linalg.Shape;

import java.util.Arrays;

public class LUPDecomp {

    private final Matrix decomp;
    private final int[] permArray;
    private Matrix lower;
    private Matrix upper;
    private Matrix perm;

    public LUPDecomp(Matrix decomp, int[] permArray) {
        this.decomp = decomp;
        this.permArray = permArray;
        this.lower = null;
        this.upper = null;
        this.perm = null;
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
            Matrix perm = new Matrix(new Shape(permArray.length, permArray.length));
            for (int i = 0; i < permArray.length; i++) {
                perm.set(i, permArray[i], 1);
            }
        }
        return perm;
    }

    public LUPDecomp deepcopy() {
        return new LUPDecomp(new Matrix(decomp), permArray == null ? null : Arrays.copyOf(permArray, permArray.length));
    }

    @Override
    public String toString() {
        return super.toString() + "\nP: " + P();
    }

}
