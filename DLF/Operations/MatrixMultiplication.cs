using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class MatrixMultiplication
    {
        public static Tensor MatMul(this Tensor A, Tensor B)
        {
            return MatrixMultiplication.Forward(A, B);
        }

        public static Tensor Forward(Tensor A, Tensor B)
        {

            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(Matrix.MatMult(A.Data, B.Data),
                    true,
                    Creators,
                    arguments: null,
                    backwardCallback: Backward);
            }

            return new Tensor(Matrix.MatMult(A.Data, B.Data));
        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            // Grad_C0 = gradient x C1.T
            self.Creators[0].Backward(gradient.MatMul(self.Creators[1].T()));

            // Grad_C1 = (gradient.T x C0).T
            self.Creators[1].Backward(gradient.T().MatMul(self.Creators[0]).T());
        }
    }

}