using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework.Layers.Initialization
{
    public class GaussianRandom : Initializator
    {
        private double mean;
        private double stdDev;
        private double height;
        private Random r;

        public GaussianRandom(double mean, double stdDev, double height, Random r)
        {
            this.r = r;
            this.mean = mean;
            this.stdDev = stdDev;
            this.height = height;
        }

        protected override double GenerateValue()
        {
            double u1 = 1.0 - r.NextDouble(); //uniform(0,1] System.Random doubles
            double u2 = 1.0 - r.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //System.Random normal(0,1)
            return (mean + stdDev * randStdNormal) * height; //System.Random normal(mean,stdDev^2)
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