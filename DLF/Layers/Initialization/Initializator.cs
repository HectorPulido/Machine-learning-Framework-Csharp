using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers.Initialization
{
    public abstract class Initializator
    {
        protected abstract double GenerateValue();
        public abstract Matrix GenerateMatrix(int x, int y);
    }
}