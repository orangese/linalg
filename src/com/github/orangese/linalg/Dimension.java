package com.github.orangese.linalg;

public class Dimension {

    private final int[] dims;

    public Dimension() {
        this.dims = new int[0];
    }

    public Dimension(Dimension shape, int newDim) {
        this.dims = new int[shape.getDims().length + 1];
        System.arraycopy(shape.getDims(), 0, this.dims, 1, shape.getDims().length);
        this.dims[0] = newDim;
    }

    public int[] getDims() {
        return dims;
    }

    public int getAlongAxis(int axis) {
        if (axis < 0) {
            axis = dims.length + axis;
        }
        try {
            return dims[axis];
        } catch (IndexOutOfBoundsException exc) {
            return axis == dims.length ? 1 : 0;
        }
    }

    public String toString() {
        StringBuilder result = new StringBuilder("(");

        for (int dim : dims) {
            result.append(dim).append(", ");
        }

        if (result.length() > 1) {
            result.setLength(result.length() - 2);
        }

        if (dims.length == 1) {
            result.append(",");
        }

        result.append(")");
        return result.toString();
    }
}
