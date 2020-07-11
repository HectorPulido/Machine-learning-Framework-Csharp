using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers.Loss {
    public class MeanSquaredError : Layer {
        public Tensor Forward (Tensor prediction, Tensor target) {
            var diff = Tensor.Sub (prediction, target);
            var mult = Tensor.Mul (diff, diff);
            return Tensor.Sum (mult, AxisZero.vertical);
        }
    }
}