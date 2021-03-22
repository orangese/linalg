import com.github.orangese.linalg.Matrix;

public class Main {

	public static void main(String[] args) {
		Matrix a = new Matrix(new double[][]{
			    new double[]{6, 0, -3,  5},
			    new double[]{4, 13, 6, -8},
			    new double[]{-1, 0, 7, 4},
			    new double[]{8, 6, 0, 2}
		});
		Matrix c = new Matrix(new double[][]{
			new double[]{-1, 9, -6},
			new double[]{7, 5, 0}
		});
		Matrix d = new Matrix(new double[][]{
			new double[]{7, 2},
			    new double[]{-4, -1}
		});
		System.out.println(a.det());
//                System.out.println(d.mul(c).subtract(a));

//                try {
//                        System.out.print("AB=");
//                        System.out.println(a.mul(b));
//                } catch (IllegalArgumentException e) {
//                        System.out.println("<impossible>");
//                }
//
//                try {
//                        System.out.print("BA=");
//                        System.out.println(b.mul(a));
//                } catch (IllegalArgumentException e) {
//                        System.out.println("<impossible>");
//                }
	}
}
