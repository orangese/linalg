package com.github.orangese.linalg;

import java.util.Objects;

public class Subspace {

    private final LUPDecomp solver;
    private final Vector[] basis;

    public Subspace(Matrix mat) {
        if (mat.size() == 0) {
            throw new IllegalArgumentException("cannot create subspace out of empty matrix");
        }
        solver = new LUPDecomp(mat);
        basis = new Vector[solver.rank()];
        for (int i = 0; i < basis.length; i++) {
            mat.copyCol(solver.getPivotPos().get(i), basis[i]);
        }
    }

    public Subspace(Vector... vectors) {
        this(new Matrix(vectors));
    }

    public boolean contains(Vector x) {
        try {
            solver.solve(x);
            return true;
        } catch (UnsupportedOperationException|IllegalArgumentException exc) {
            return false;
        }
    }

    public Vector[] basis() {
        return basis;
    }

    public int dim() {
        return basis().length;
    }

    public int encompassingSpace() {
        return solver.U().rowDim();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Subspace subspace = (Subspace) o;
        return Objects.equals(solver, subspace.solver);
    }

    @Override
    public int hashCode() {
        return Objects.hash(solver);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("Basis: ");
        for (Vector vec : basis()) {
            result.append(vec).append(",\n");
        }
        result.setLength(result.length() - 2);
        return result.toString();
    }
}
