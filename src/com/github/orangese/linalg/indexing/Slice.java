package com.github.orangese.linalg.indexing;

import com.github.orangese.linalg.Dimension;

public class Slice {

    private Slice[] slices;
    private int start;
    private int stop;

    public Slice(int start) {
        this(start, -1);
    }

    public Slice(int start, int stop) {
        this.start = start;
        this.stop = stop;
    }

    public Slice(Slice ... slices) {
        this.slices = slices;
    }

    public int getStart() {
        return start;
    }

    public int getStop() {
        return stop;
    }

    public Slice[] getSubSlices() {
        return slices;
    }

    public boolean isRange() {
        return slices == null;
    }

    public void checkRange(int newStop, int length) {
        if (newStop > length) {
            throw new IndexOutOfBoundsException("index " + newStop + " out of bounds for axis with length " + length);
        }
    }

    public int calcNewStart(Dimension shape) {
        return getStart() < 0 ? shape.getAlongAxis(0) + getStart() : getStart();
    }

    public int calcNewStop(Dimension shape) {
        return getStop() < 0 ? shape.getAlongAxis(0) + getStop() : getStop();
    }

    public String toString() {
        StringBuilder result = new StringBuilder("[");
        if (slices == null) {
            return result.append(start).append(":").append(stop).append("]").toString();
        } else {
            for (Slice slice : slices) {
                String sliceStr = slice.toString();
                result.append(sliceStr, 1, sliceStr.length() - 1).append(", ");
            }
            result.setLength(result.length() - 2);
            return result.append("]").toString();
        }
    }
}
