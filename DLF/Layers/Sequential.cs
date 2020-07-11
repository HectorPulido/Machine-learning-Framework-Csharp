using System;
using System.Collections.Generic;

namespace DLFramework.Layers {
    class Sequential : Layer {
        private List<Layer> layers;

        public List<Layer> Layers { get => layers; }

        public override List<Tensor> Parameters { get => GetParameters (); }

        public Sequential (List<Layer> layers) {
            this.layers = layers;
        }

        public Sequential () {
            layers = new List<Layer> ();
        }

        public override Tensor Forward (Tensor input) {
            foreach (var layer in layers) {
                input = layer.Forward (input);
            }
            return input;
        }

        public List<Tensor> GetParameters () {
            List<Tensor> temp = new List<Tensor> ();
            foreach (var layer in layers) {
                temp.AddRange(layer.Parameters);
            }
            return temp;
        }
    }
}