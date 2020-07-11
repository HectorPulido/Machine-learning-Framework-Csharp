using System;
using System.Collections.Generic;

namespace DLFramework.Layers
{
    public class Layer
    {
        protected List<Tensor> parameters;
        public virtual List<Tensor> Parameters { get => parameters; set => parameters = value; }

        public Layer()
        {
            parameters = new List<Tensor>();
        }

        public virtual Tensor Forward(Tensor input)
        {
            return null;
        }
    }
}