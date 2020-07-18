using LinearAlgebra;
using DLFramework.Operations;

namespace DLFramework.Layers.Loss {
    public class MeanSquaredError : Layer {
        public Tensor Forward (Tensor prediction, Tensor target) {
            var diff = prediction.Sub(target);
            var mult = diff.Mul(diff);
            return mult.Sum(AxisZero.vertical);
        }
    }
}