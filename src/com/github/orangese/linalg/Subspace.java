package com.github.orangese.linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Subspace {

    private final LUPDecomp solver;

    public Subspace() {
        solver = new LUPDecomp(new Matrix(new Shape(0, 0)));
    }

    public Subspace(Vector... vectors) {
        if (vectors.length < 1) {
            throw new IllegalArgumentException("at least one vector needs to be provided");
        }

        List<Integer> pivotPos = new LUPDecomp(new Matrix(vectors)).getPivotPos();
        List<Vector> basisVectors = new ArrayList<>();

        for (Integer pos : pivotPos) {
            basisVectors.add(vectors[pos]);
        }

        solver = new LUPDecomp(new Matrix(basisVectors.toArray(new Vector[0])));
    }

    public boolean contains(Vector vec) {
        try {
            solver.solve(vec);
            return true;
        } catch (UnsupportedOperationException|IllegalArgumentException exc) {
            exc.printStackTrace();
            return false;
        }
    }

    public Vector[] basis() {
        Vector[] basis = new Vector[solver.U().rowDim()];
        for (int i = 0; i < basis.length; i++) {
            basis[i] = solver.U().getRow(i);
        }
        return basis;
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
