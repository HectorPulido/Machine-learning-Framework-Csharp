using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Substraction
    {
        public static Tensor Sub(this Tensor A, Tensor B)
        {
            return Substraction.Forward(A, B);
        }

        public static Tensor Forward(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(A.Data - B.Data,
                    true,
                    Creators,
                    arguments: null,
                    backwardCallback: Backward);
            }

            return new Tensor(A.Data - B.Data);
        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            // Grad_C0 = gradient
            self.Creators[0].Backward(gradient, self);
            // Grad_C1 = -gradient
            self.Creators[1].Backward(Negation.Forward(gradient), self);
        }
    }

}