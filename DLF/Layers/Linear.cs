using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers {
    public class Linear : Layer {
        public Linear (int input, int output, Random r) {
            var w = (Matrix.Random (input, output, r) * 2) - 1;
            var weights = new Tensor (w, true);
            var bias = new Tensor (Matrix.Zeros (1, output), true);
            parameters.Add (weights);
            parameters.Add (bias);
        }
        public override Tensor Forward (Tensor input) {
            //out = (weights x input) + bias.expanded
            var bias = Tensor.Expand (parameters[1], AxisZero.vertical, input.Data.X);
            //System.Console.WriteLine (Tensor.MatMul (input, parameters[0]).Data.Size);
            System.Console.WriteLine (bias.Data.Size);
            System.Console.WriteLine (input.Data.Size);
            System.Console.WriteLine (parameters[0].Data.Size);
            return Tensor.Add (Tensor.MatMul (input, parameters[0]), bias);
        }
    }
}