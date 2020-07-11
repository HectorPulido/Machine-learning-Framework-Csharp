using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers.Initialization
{
    public class UniformRandom : Initializator
    {
        private double min;
        private double max;
        private Random r;

        public UniformRandom(double min, double max, Random r)
        {
            this.r = r;
            this.max = max;
            this.min = min;
        }

        protected override double GenerateValue()
        {
            return r.NextDouble() * (min - max) + max;
        }

        public override Matrix GenerateMatrix(int x, int y)
        {
            var matrix = (Matrix)new double[x, y];
            Matrix.MatrixLoop((i, j) =>
            {
                matrix[i, j] += this.GenerateValue();
            }, matrix.X, matrix.Y);
            return matrix;
        }
    }
}