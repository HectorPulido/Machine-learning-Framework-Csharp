using System;
using System.Collections.Generic;
using DLFramework;
using DLFramework.Layers;
using DLFramework.Layers.Activation;
using DLFramework.Operations;
using DLFramework.Layers.Loss;
using DLFramework.Optimizers;
using LinearAlgebra;

class MainClass {
    public static void Main (string[] args) {
        FifthNN ();

        /*        var test = new Tensor ((Matrix) new double[, ] { {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 } }, true);
                var t1 = Sigmoid.Forward (test);
                Console.WriteLine ($"Sigmoid: {t1}");
                t1.Backward(new Tensor (Matrix.Ones (test.Data.X, test.Data.Y)));*/

    }

    static void FifthNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 1 }, { 0 }, { 0 }, { 1 } }, true);

        var seq = new Sequential ();
        seq.Layers.Add (new Linear (2, 5, r));
        seq.Layers.Add (new SigmoidLayer ());
        seq.Layers.Add (new Linear (5, 5, r));
        seq.Layers.Add (new SigmoidLayer ());
        seq.Layers.Add (new Linear (5, 1, r));
        seq.Layers.Add (new SigmoidLayer ());

        var sgd = new StochasticGradientDescent (seq.Parameters, 1f);

        var mse = new MeanSquaredError ();

        for (var i = 0; i < 300; i++) {
            var pred = seq.Forward (data);

            var loss = mse.Forward (pred, target);

            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
            sgd.Step ();

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }
    static void FourthNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 0 }, { 1 }, { 0 }, { 1 } }, true);

        var seq = new Sequential ();
        seq.Layers.Add (new Linear (2, 3, r));
        seq.Layers.Add (new Linear (3, 1, r));

        var sgd = new StochasticGradientDescent (seq.Parameters, 0.1f);

        var mse = new MeanSquaredError ();

        for (var i = 0; i < 10; i++) {
            var pred = seq.Forward (data);

            var loss = mse.Forward (pred, target);

            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
            sgd.Step ();

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }

    static void ThirdNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 0 }, { 1 }, { 0 }, { 1 } }, true);

        var seq = new Sequential ();
        seq.Layers.Add (new Linear (2, 3, r));
        seq.Layers.Add (new Linear (3, 1, r));

        var sgd = new StochasticGradientDescent (seq.Parameters, 0.1f);

        for (var i = 0; i < 10; i++) {
            var pred = seq.Forward (data);

            var diff = pred.Sub(target);
            var loss = diff.Mul(diff).Sum(AxisZero.vertical);

            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
            sgd.Step ();

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }

    static void SecondNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 0 }, { 1 }, { 0 }, { 1 } }, true);

        var weights = new List<Tensor> ();
        weights.Add (new Tensor (Matrix.Random (2, 3, r), true));
        weights.Add (new Tensor (Matrix.Random (3, 1, r), true));

        var sgd = new StochasticGradientDescent (weights, 0.1f);

        for (var i = 0; i < 10; i++) {
            var pred = data.MatMul(weights[0]).Add(weights[1]);

            var diff = pred.Sub(target);
            var loss = diff.Mul(diff).Sum(AxisZero.vertical);

            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
            sgd.Step ();

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }

    static void FirstNN () {
        var r = new Random ();

        var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
        var target = new Tensor ((Matrix) new double[, ] { { 0 }, { 1 }, { 0 }, { 1 } }, true);

        var weights = new List<Tensor> ();
        weights.Add (new Tensor (Matrix.Random (2, 3, r), true));
        weights.Add (new Tensor (Matrix.Random (3, 1, r), true));

        for (var i = 0; i < 10; i++) {
            var pred = data.MatMul(weights[0]).Add(weights[1]);

            var diff = pred.Sub(target);
            var loss = diff.Mul(diff).Sum(AxisZero.vertical);
            loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));

            foreach (var weight in weights) {
                weight.Data -= (weight.Gradient.Data * 0.1f);
                weight.Gradient.Data *= 0f;
            }

            Console.WriteLine ($"Epoch: {i} Loss: {loss}");
        }
    }

}