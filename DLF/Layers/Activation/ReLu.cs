using System;
using System.Collections.Generic;
using LinearAlgebra;
using DLFramework.Operations;

namespace DLFramework.Layers.Activation
{

    public class ReLu
    {
        public static Tensor Forward(Tensor input)
        {
            double[,] output = input.Data;
            Matrix.MatrixLoop((i, j) =>
            {
                output[i, j] = output[i, j] > 0 ? output[i, j] : 0;
            }, input.Data.X, input.Data.Y);

            if (input.AutoGrad)
            {
                var Creators = new List<Tensor>() { input };
                return new Tensor(
                    data: output,
                    autoGrad: true,
                    creators: Creators,
                    arguments: null,
                    backwardCallback: ReLu.Backward);
            }

            return new Tensor(output);
        }
        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            var ones = new Tensor(Matrix.Ones(gradient.Data.X, gradient.Data.Y));
            double[,] derivative = self.Data;
            Matrix.MatrixLoop((i, j) =>
            {
                derivative[i, j] = derivative[i, j] > 0 ? 1 : 0;
            }, self.Data.X, self.Data.Y);

            var derivatives = new Tensor(derivative);
            creators[0].Backward(gradient.Mul(derivatives));
        }
    }

}