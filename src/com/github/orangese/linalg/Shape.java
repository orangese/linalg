package com.github.orangese.linalg;

import java.util.Objects;

public class Shape {

    private final int rowDim;
    private final int colDim;

    public Shape() {
        this(0, 0);
    }

    public Shape(int rowDim, int colDim) {
        this.rowDim = rowDim;
        this.colDim = colDim;
    }

    public Shape(Shape shape) {
        this.rowDim = shape.rowDim;
        this.colDim = shape.colDim;
    }

    public int rowDim() {
        return rowDim;
    }

    public int colDim() {
        return colDim;
    }

    public int size() {
        return Math.max(rowDim * colDim, 1);
    }

    public int ndims() {
        return (this.rowDim == 0 ? 0: 1) + (this.colDim == 0 ? 0: 1);
    }

    public boolean equals(int rowDim, int colDim) {
        return equals(new Shape(rowDim, colDim));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return rowDim == shape.rowDim &&
                colDim == shape.colDim;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rowDim, colDim);
    }

    public String toString() {
        return "(" + rowDim + ", " + colDim + ")";
    }

}
