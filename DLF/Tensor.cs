using System;
using System.Collections.Generic;
using LinearAlgebra;
using DLFramework.Operations;

namespace DLFramework
{
    public class Tensor
    {
        private static int idCount = 0;

        private Matrix data;
        private List<Tensor> creators;
        private Dictionary<int, int> childrens;
        private Tensor gradient;
        private bool autoGrad;
        private int id;
        private List<object> arguments;
        private Action<Tensor, Tensor, List<Tensor>> backwardCallback;

        public Matrix Data { get => data; set => data = value; }

        public List<Tensor> Creators { get => creators; }
        public Tensor Gradient { get => gradient; set => gradient = value; }
        public bool AutoGrad { get => autoGrad; }
        public int Id { get => id; }
        public Dictionary<int, int> Childrens { get => childrens; set => childrens = value; }
        public List<object> Arguments { get => arguments; }

        public Tensor(Matrix data,
            bool autoGrad = false,
            List<Tensor> creators = null,
            List<object> arguments = null,
            Action<Tensor, Tensor, List<Tensor>> backwardCallback = null)
        {
            this.data = data;
            this.autoGrad = autoGrad;
            this.gradient = null;
            this.arguments = arguments;
            this.backwardCallback = backwardCallback;

            //Unique id
            id = idCount;
            idCount++;

            //Child and parents
            this.creators = creators;
            childrens = new Dictionary<int, int>();

            if (this.creators != null)
            {
                foreach (var creator in this.creators)
                {
                    if (creator.Childrens.ContainsKey(id))
                    {
                        creator.Childrens[id] += 1;
                    }
                    else
                    {
                        creator.Childrens.Add(id, 1);
                    }
                }
            }
        }

        private bool allChildrenGradsAccountedFor()
        {
            foreach (var child in childrens)
            {
                if (child.Value != 0)
                {
                    return false;
                }
            }
            return true;
        }

        public void Backward(Tensor gradient = null, Tensor gradientOrigin = null)
        {
            if (!autoGrad)
            {
                //                throw new ArgumentException($"This tensor is not set as autograd");
                return;
            }

            if (gradient == null)
            {
                gradient = new Tensor(Matrix.Ones(data.X, data.Y));
            }

            if (gradientOrigin != null)
            {
                if (childrens[gradientOrigin.Id] == 0)
                {
                    throw new ArgumentException($"Cannot backprop more than once");
                }
                childrens[gradientOrigin.Id] -= 1;
            }

            if (this.gradient == null)
            {
                this.gradient = gradient;
            }
            else
            {
                this.gradient = this.gradient.Add(gradient);
            }

            // Console.WriteLine ("===============BACKPROP DATA======================");
            // Console.WriteLine ($"this id: {this.id}");
            // Console.WriteLine ($"Operation: {this.creationOperation}");
            // Console.WriteLine ("==================================================");

            if (creators != null && (allChildrenGradsAccountedFor() || gradientOrigin == null))
            {
                backwardCallback(this, this.gradient, creators);
             }
        }

        public override string ToString()
        {
            return data.ToString();
        }

        //==============BACKPROPAGATION===================
        private void CheckCreatorsThrow(int creatorNumber)
        {
            if (Creators == null)
            {
                throw new ArgumentException($"Creators can not be null");
            }

            if (Creators.Count != creatorNumber)
            {
                throw new ArgumentException($"Creator count must be 2 not {Creators.Count}");
            }
        }

        private void CheckArgumentsThrow(int argumentsNumber)
        {
            if (arguments == null)
            {
                throw new ArgumentException($"Arguments are null");
            }

            if (arguments.Count != argumentsNumber)
            {
                throw new ArgumentException($"Number of arguments must be { argumentsNumber }");
            }
        }

        private bool CheckCreators(int creatorNumber)
        {
            if (Creators == null)
            {
                return false;
            }

            if (Creators.Count != creatorNumber)
            {
                return false;
            }

            return true;
        }
    }
}