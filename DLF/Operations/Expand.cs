using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Expand
    {
        public static Tensor Exp(this Tensor A, AxisZero dim, int copies, int copies2 = 0)
        {
            return Expand.Forward(A, dim, copies, copies2);
        }

        public static Tensor Forward(Tensor A, AxisZero dim, int copies, int copies2 = 0)
        {
            Matrix m = null;
            if (dim == AxisZero.horizontal)
            {
                m = Matrix.Zeros(A.Data.X, copies);
                Matrix.MatrixLoop((i, j) =>
                {
                    m[i, j] = A.Data[i, 0];
                }, A.Data.X, copies);
            }
            else if (dim == AxisZero.vertical)
            {
                m = Matrix.Zeros(copies, A.Data.Y);
                Matrix.MatrixLoop((i, j) =>
                {
                    m[i, j] = A.Data[0, j];
                }, copies, A.Data.Y);
            }
            else if (dim == AxisZero.none)
            {
                m = Matrix.Zeros(copies, copies2);
                Matrix.MatrixLoop((i, j) =>
                {
                    m[i, j] = A.Data[0, 0];
                }, copies, copies2);
            }

            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                var Argument = new List<object>() { dim };
                return new Tensor(m,
                    autoGrad: true,
                    creators: Creators,
                    arguments: Argument,
                    backwardCallback: Backward
                );
            }

            return new Tensor(m);
        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            var dimension = (AxisZero)self.Arguments[0];
            self.Creators[0].Backward(gradient.Sum(dimension));
        }
    }

}