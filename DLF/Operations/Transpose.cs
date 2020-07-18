using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Transpose
    {
        public static Tensor T(this Tensor A)
        {
            return Transpose.Forward(A);
        }

        public static Tensor Forward(Tensor A)
        {

            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.Data.T,
                    true,
                    Creators,
                    arguments: null,
                    backwardCallback: Transpose.Backward);
            }

            return new Tensor(A.Data.T);

        }

        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            self.Creators[0].Backward(gradient.T());
        }
    }

}