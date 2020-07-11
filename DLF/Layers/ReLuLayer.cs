using LinearAlgebra;
using DLFramework.Layers.Activation;

namespace DLFramework.Layers {
    public class ReLuLayer : Layer {
        public override Tensor Forward (Tensor input) {
            return ReLu.Forward (input);
        }
    }
}