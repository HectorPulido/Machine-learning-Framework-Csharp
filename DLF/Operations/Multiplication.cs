using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Multiplication
    {
        public static Tensor Mul(this Tensor A, Tensor B)
        {
            return Multiplication.Forward(A, B);
        }

        public static Tensor Forward(Tensor A, Tensor B)
        {

            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(Matrix.DeltaMult(A.Data, B.Data),
                    true,
                    Creators,
                    arguments: null,
                    backwardCallback: Backward);
            }

            return new Tensor(Matrix.DeltaMult(A.Data, B.Data));
        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            // Grad_C0 = gradient . C1
            self.Creators[0].Backward(gradient.Mul(self.Creators[1]), self);
            // Grad_C1 = gradient . C0
            self.Creators[1].Backward(gradient.Mul(self.Creators[0]), self);
        }
    }

}