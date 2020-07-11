using System;
using System.Collections.Generic;
using LinearAlgebra;
using DLFramework.Layers.Initialization;

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
            var inputXWeights = Tensor.MatMul(input, parameters[0]);
            var bias = Tensor.Expand(parameters[1], AxisZero.vertical, inputXWeights.Data.X);
            return Tensor.Add(inputXWeights, bias);
        }
    }
}