using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers.Loss {
    public class MeanSquaredError : Layer {
        public Tensor Forward (Tensor prediction, Tensor target) {
            var diff = Tensor.Sub (prediction, target);
            return Tensor.Sum (Tensor.Mul (diff, diff), AxisZero.vertical);
        }
    }
}