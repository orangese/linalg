package com.github.orangese.linalg.decomp;

import com.github.orangese.linalg.Matrix;

public class LUDecomp {

    private final Matrix L;
    private final Matrix U;

    public LUDecomp(Matrix L, Matrix U) {
        this.L = L;
        this.U = U;
    }

    public Matrix L() {
        return L;
    }

    public Matrix U() {
        return U;
    }

    public LUDecomp deepcopy() {
        return new LUDecomp(new Matrix(L), new Matrix(U));
    }

    @Override
    public String toString() {
        return  "L: " + L + "\nU: " + U;
    }
}
