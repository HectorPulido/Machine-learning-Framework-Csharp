using System;
using System.Collections.Generic;
using LinearAlgebra;
using DLFramework.Operations;

namespace DLFramework.Layers.Activation {

    public class Sigmoid {
        public static Tensor Forward (Tensor input) {
            double[, ] output = input.Data;
            Matrix.MatrixLoop ((i, j) => {
                output[i, j] = 1 / (1 + Math.Exp (-output[i, j]));

            }, input.Data.X, input.Data.Y);

            if (input.AutoGrad) {
                var Creators = new List<Tensor> () { input };
                return new Tensor (
                    data: output,
                    autoGrad: true,
                    creators: Creators,
                    arguments: null,
                    backwardCallback: Sigmoid.Backward);
            }

            return new Tensor (output);
        }
        public static void Backward (Tensor self, Tensor gradient, List<Tensor> creators) {
            var ones = new Tensor (Matrix.Ones (gradient.Data.X, gradient.Data.Y));
            //backward = grad * (self * (ones - self)))
            double[, ] derivative = self.Data;
            Matrix.MatrixLoop ((i, j) => {
                //double sig = 1.0 / (1.0 + Math.Exp (-derivative[i, j]));
                derivative[i, j] = derivative[i, j] * (1.0 - derivative[i, j]);
            }, self.Data.X, self.Data.Y);

            var derivatives = new Tensor (derivative); //Tensor.Mul (self, Tensor.Sub (ones, self));
            creators[0].Backward (gradient.Mul(derivatives));
        }
    }

}