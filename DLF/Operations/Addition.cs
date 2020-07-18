using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Operations
{

    public static class Addition
    {

        public static Tensor Add(this Tensor A, Tensor B)
        {
            return Addition.Forward(A, B);
        }

        public static Tensor Forward(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(
                        data: A.Data + B.Data,
                        autoGrad: true,
                        creators: Creators,
                        arguments: null,
                        backwardCallback: Backward);

            }

            return new Tensor(A.Data + B.Data);
        }


        public static void Backward(Tensor self, Tensor gradient, List<Tensor> creators)
        {
            self.Creators[0].Backward(gradient, self);
            self.Creators[1].Backward(gradient, self);
        }
    }

}