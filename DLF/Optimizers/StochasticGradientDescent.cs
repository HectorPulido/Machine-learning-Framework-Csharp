using System.Collections.Generic;

namespace DLFramework.Optimizers {
    public class StochasticGradientDescent {

        private float alpha;
        private List<Tensor> parameters;
        public List<Tensor> Parameters { get => parameters; set => parameters = value; }
        public float Alpha { get => alpha; }

        public StochasticGradientDescent (List<Tensor> parameters, float alpha = 0.1f) {
            this.parameters = parameters;
            this.alpha = alpha;
        }

        public void Zero () {
            foreach (var parameter in parameters) {
                parameter.Gradient.Data *= 0f;
            }
        }

        public void Step (bool zero = true) {
            foreach (var parameter in parameters) {
                parameter.Data -= parameter.Gradient.Data * alpha;
                if (zero) {
                    parameter.Gradient.Data *= 0;
                }
            }
        }

    }
}