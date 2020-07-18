using System;
using LinearAlgebra;
using DLFramework.Layers.Initialization;
using DLFramework.Operations;

namespace DLFramework.Layers
{
    public class Linear : Layer
    {
        public Linear(int input, int output, Random r, Initializator init = null)
        {
            if (init == null)
            {
                init = new GaussianRandom(0, 1, 1, r);
            }

            var w = init.GenerateMatrix(input, output);
            var b = init.GenerateMatrix(1, output);
            var weights = new Tensor(w, true);
            var bias = new Tensor(b, true);
            parameters.Add(bias);
            parameters.Add(weights);
        }
        public override Tensor Forward(Tensor input)
        {
            //out = (input x weights) + bias.expanded
            var inputXWeights = input.MatMul(parameters[1]);
            var bias = parameters[0].Exp(AxisZero.vertical, inputXWeights.Data.X);
            return inputXWeights.Add(bias);
        }
    }
}