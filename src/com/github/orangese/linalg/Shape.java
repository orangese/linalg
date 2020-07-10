package com.github.orangese.linalg;

import java.util.Arrays;
import java.util.Objects;

public class Shape {

    private final int[] shape;
    public final int length;

    public Shape(int... shape) {
        this.shape = shape;
        this.length = shape.length;
    }

    public Shape(Shape shape) {
        this.shape = new int[shape.length];
        this.length = shape.length;
        System.arraycopy(shape.shape, 0, this.shape, 0, length);
    }

    public int axis(int axis) {
        return shape[axis];
    }

    public int[] toArray() {
        return shape;
    }

    public boolean equals(int... other) {
        return equals(new Shape(other));
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        Shape otherShape = (Shape) other;
        return length == otherShape.length && Arrays.equals(shape, otherShape.shape);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(length);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    public String toString() {
        StringBuilder result = new StringBuilder("(");
        for (int subShape : shape) {
            result.append(subShape).append(", ");
        }
        result.setLength(result.length() - 2);
        return result.append(")").toString();
    }

}
