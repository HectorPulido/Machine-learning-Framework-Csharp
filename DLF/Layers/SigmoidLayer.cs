using LinearAlgebra;
using DLFramework.Layers.Activation;

namespace DLFramework.Layers {
    public class SigmoidLayer : Layer {
        public override Tensor Forward (Tensor input) {
            return Sigmoid.Forward (input);
        }
    }
}