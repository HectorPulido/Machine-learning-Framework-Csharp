using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Summation
    {
        public static Tensor Sum(this Tensor A, AxisZero dim)
        {
            return Summation.Forward(A, dim);
        }

        public static Tensor Forward(Tensor A, AxisZero dim)
        {
            if (A.AutoGrad)
            {
                var Argument = new List<object>() { dim, new int[] { A.Data.X, A.Data.Y } };
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.Data.Sumatory(dim),
                    autoGrad: true,
                    creators: Creators,
                    arguments: Argument,
                    backwardCallback: Backward);
            }

            return new Tensor(A.Data.Sumatory(dim));
        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            var dimension = (AxisZero)self.Arguments[0];
            var shape = new int[] { };
            var copies = 0;

            if (dimension == AxisZero.horizontal)
            {
                copies = (int)self.Creators[0].Data.Y;
            }
            else if (dimension == AxisZero.vertical)
            {
                copies = (int)self.Creators[0].Data.X;
            }
            else
            {
                shape = (int[])self.Arguments[1];
                self.Creators[0].Backward(
                    gradient.Exp(dimension, shape[0], shape[1])
                );
                return;
            }

            self.Creators[0].Backward(
                gradient.Exp(dimension, copies)
            );
        }
    }

}