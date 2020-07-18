using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Negation
    {
        public static Tensor Neg(this Tensor A)
        {
            return Negation.Forward(A);
        }

        public static Tensor Forward(Tensor A)
        {
            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.Data * -1,
                    true,
                    Creators,
                    arguments: null,
                    backwardCallback: Backward);
            }

            return new Tensor(A.Data * -1);
        }


        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            self.Creators[0].Backward(Negation.Forward(gradient), self);
        }
    }

}