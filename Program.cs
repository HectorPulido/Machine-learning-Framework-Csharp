using System;
using System.Collections.Generic;
using DLFramework;
using LinearAlgebra;

class MainClass {
    public static void Main (string[] args) {
        //TestExpand();
        TestNN ();
    }

    static void TestNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 0 }, { 1 }, { 0 }, { 1 } }, true);

        var weights = new List<Tensor> ();
        weights.Add (new Tensor (Matrix.Random (2, 3, r), true));
        weights.Add (new Tensor (Matrix.Random (3, 1, r), true));

        for (var i = 0; i < 10; i++) {
            var pred = Tensor.MatMul (Tensor.MatMul (data, weights[0]), weights[1]);

            var diff = Tensor.Sub (pred, target);
            var loss = Tensor.Sum (Tensor.Mul (diff, diff), AxisZero.vertical);

            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));

            foreach (var weight in weights) {
                weight.Data -= (weight.Gradient.Data * 0.1f);
                weight.Gradient.Data *= 0f;
            }

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }

    static void TestExpand () {
        var data = new Tensor ((Matrix) new double[, ] { { 1 } });
        Console.WriteLine ($"Shape weight {Tensor.Expand(data, AxisZero.horizontal, 4).Data}");
    }

    static void Test1 () {
        //TEST

        var x = new Tensor ((Matrix) new double[, ] { { 1, 2, 3, 4, 5 } }, true);
        var y = new Tensor ((Matrix) new double[, ] { { 1, 1, 1, 1, 1 } }, true);

        var z = Tensor.Add (x, y);

        //Sum
        Console.WriteLine ("Matrix a + 1, d");
        Console.WriteLine (z.ToString ());

        //Backward
        Console.WriteLine ("Grad x before");
        Console.WriteLine (x.Gradient);

        Console.WriteLine ("Grad y before");
        Console.WriteLine (y.Gradient);

        z.Backward (new Tensor ((Matrix) new double[, ] { { 1, 1, 1, 1, 1 } }));
        Console.WriteLine ("Creators");
        foreach (var creator in z.Creators) {
            Console.WriteLine (creator.ToString ());
        }
        Console.WriteLine ("Operation");
        Console.WriteLine (z.CreationOperation);

        Console.WriteLine ("Grad x");
        Console.WriteLine (x.Gradient);

        Console.WriteLine ("Grad y");
        Console.WriteLine (y.Gradient);

        //Backward Stress
        var a = new Tensor ((Matrix) new double[, ] { { 1, 2, 3, 4, 5 } }, true);
        var b = new Tensor ((Matrix) new double[, ] { { 2, 2, 2, 2, 2 } }, true);
        var c = new Tensor ((Matrix) new double[, ] { { 5, 4, 3, 2, 1 } }, true);

        var d = Tensor.Add (a, b);
        var e = Tensor.Add (b, c);

        var f = Tensor.Add (d, e);

        f.Backward (new Tensor ((Matrix) new double[, ] { { 1, 1, 1, 1, 1 } }));
        Console.WriteLine ("Grad b");
        Console.WriteLine (b.Gradient);

        //Negation
        var a2 = new Tensor ((Matrix) new double[, ] { { 1, 2, 3, 4, 5 } }, true);
        var b2 = new Tensor ((Matrix) new double[, ] { { 2, 2, 2, 2, 2 } }, true);
        var c2 = new Tensor ((Matrix) new double[, ] { { 5, 4, 3, 2, 1 } }, true);

        var d2 = Tensor.Add (a2, Tensor.Neg (b2));
        var e2 = Tensor.Add (Tensor.Neg (b2), c2);

        var f2 = Tensor.Add (d2, e2);
        f2.Backward (new Tensor ((Matrix) new double[, ] { { 1, 1, 1, 1, 1 } }));

        Console.WriteLine ("Grad b2");
        Console.WriteLine (b2.Gradient);
    }
}