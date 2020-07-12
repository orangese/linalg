package com.github.orangese.linalg.decomp;

import com.github.orangese.linalg.Matrix;

public class LUPDecomp extends LUDecomp {

    private final Matrix P;

    public LUPDecomp(Matrix P, Matrix L, Matrix U) {
        super(L, U);
        this.P = P;
    }

    public Matrix P() {
        return P;
    }

    @Override
    public LUPDecomp deepcopy() {
        return new LUPDecomp(new Matrix(P), new Matrix(L()), new Matrix(U()));
    }

    @Override
    public String toString() {
        return super.toString() + "\nP: " + P;
    }

}
