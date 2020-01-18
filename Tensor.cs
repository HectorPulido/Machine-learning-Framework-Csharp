using System;
using System.Collections.Generic;
using LinearAlgebra;

namespace DLFramework
{
    public class Tensor
    {
        private static int idCount = 0;

        private Matrix data;
        private List<Tensor> creators;
        private Dictionary<int, int> childrens;
        private TensorOperations creationOperation;
        private Tensor gradient;
        private bool autoGrad;
        private int id;
        private List<object> arguments;

        public Matrix Data { get => data; set => data = value; }
        public List<Tensor> Creators { get => creators; set => creators = value; }
        public TensorOperations CreationOperation { get => creationOperation; set => creationOperation = value; }
        public Tensor Gradient { get => gradient; set => gradient = value; }
        public bool AutoGrad { get => autoGrad; set => autoGrad = value; }
        public int Id { get => id; set => id = value; }
        public Dictionary<int, int> Childrens { get => childrens; set => childrens = value; }
        public List<object> Arguments { get => arguments; set => arguments = value; }

        public Tensor(Matrix data,
            bool autoGrad = false,
            List<Tensor> creators = null,
            TensorOperations creationOperation = TensorOperations.None,
            List<object> arguments = null)
        {
            this.data = data;
            this.autoGrad = autoGrad;
            this.gradient = null;
            this.arguments = arguments;

            //Unique id
            id = idCount;
            idCount++;

            //Child and parents
            this.creators = creators;
            this.creationOperation = creationOperation;
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

        public void Backward(Tensor gradient, Tensor gradientOrigin = null)
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

            if (gradient == null)
            {
                gradient = gradient;
            }
            else
            {
                gradient = Tensor.Add(gradient, gradient);
            }

            if (data.X != gradient.data.X || data.Y != gradient.data.Y)
            {
                Console.WriteLine("=====================================");
                Console.WriteLine($"this id: {this.id}");
                Console.WriteLine($"Shape data: {this.data.Size}");
                Console.WriteLine($"Shape gradient: {this.gradient.data.Size}");
                Console.WriteLine($"Operation: {this.creationOperation}");
                Console.WriteLine("=====================================");
            }

            if (creators != null && (allChildrenGradsAccountedFor() || gradientOrigin == null))
            {
                switch (creationOperation)
                {
                    case TensorOperations.None:
                        //Do nothing
                        break;
                    case TensorOperations.Addition:
                        AdditionTensorOperation();
                        break;
                    case TensorOperations.Negation:
                        NegationTensorOperation();
                        break;
                    case TensorOperations.Substraction:
                        SubstractionTensorOperation();
                        break;
                    case TensorOperations.Multiplication:
                        MultiplicationTensorOperation();
                        break;
                    case TensorOperations.Sumatory:
                        SumatoryTensorOperation();
                        break;
                    case TensorOperations.Transpose:
                        TransposeTensorOperation();
                        break;
                    case TensorOperations.MatrixMultiplication:
                        MatrixMultiplicationTensorOperation();
                        break;
                    case TensorOperations.Expand:
                        ExpandTensorOperation();
                        break;
                    default:
                        throw new ArgumentException($"Invalid Creation operation: {creationOperation}");
                }
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

        private void SumatoryTensorOperation()
        {
            CheckCreatorsThrow(1);
            CheckArgumentsThrow(1);
            var dimension = (AxisZero)arguments[0];
            var copies = 0;

            if (dimension == AxisZero.horizontal)
            {
                copies = (int)Creators[0].data.Y;
            }
            else
            {
                copies = (int)Creators[0].data.X;
            }

            Console.WriteLine(Creators[0].data.Size);

            Console.WriteLine("=====================================");
            Console.WriteLine(gradient);
            Creators[0].Backward(Tensor.Expand(gradient, dimension, copies));

        }

        private void MatrixMultiplicationTensorOperation()
        {
            var newGrad = Tensor.MatMul(gradient, Tensor.Transp(Creators[1]));
            Creators[0].Backward(newGrad);

            var newGrad2 = Tensor.Transp(Tensor.MatMul(Tensor.Transp(gradient), Creators[0]));
            Creators[1].Backward(newGrad);
        }

        private void TransposeTensorOperation()
        {
            CheckCreatorsThrow(1);
            Creators[0].Backward(Tensor.Transp(gradient));
        }

        private void MultiplicationTensorOperation()
        {
            CheckCreatorsThrow(2);
            var newGrad = Tensor.Mul(gradient, Creators[1]);
            Creators[0].Backward(newGrad, this);
            var newGrad2 = Tensor.Mul(gradient, Creators[0]);
            Creators[1].Backward(newGrad2, this);
        }

        private void SubstractionTensorOperation()
        {
            CheckCreatorsThrow(2);
            Creators[0].Backward(gradient, this);
            Creators[1].Backward(Tensor.Neg(gradient), this);
        }

        private void NegationTensorOperation()
        {
            CheckCreatorsThrow(1);
            Creators[0].Backward(Tensor.Neg(gradient), this);
        }

        private void AdditionTensorOperation()
        {
            CheckCreatorsThrow(2);
            Creators[0].Backward(gradient, this);
            Creators[1].Backward(gradient, this);
        }

        private void ExpandTensorOperation()
        {
            CheckCreatorsThrow(1);
            CheckArgumentsThrow(1);
            var dimension = (AxisZero)arguments[0];
            Creators[0].Backward(Tensor.Sum(gradient, dimension));
        }

        //==============OPERATIONS===================
        public static Tensor Expand(Tensor A, AxisZero dim, int copies)
        {
            Matrix m = null;
            if (dim == AxisZero.horizontal)
            {
                m = Matrix.Zeros(A.data.X, copies);
                Matrix.MatrixLoop((i, j) =>
                {
                    m[i, j] = A.data[i, 0];
                }, A.data.X, copies);
            }
            else if (dim == AxisZero.vertical)
            {
                m = Matrix.Zeros(copies, A.data.Y);
                Matrix.MatrixLoop((i, j) =>
                {
                    m[i, j] = A.data[0, j];
                }, copies, A.data.Y);
            }
            Console.WriteLine(m);
            Console.WriteLine("=====================================");
            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                var Argument = new List<object>() { dim };
                return new Tensor(m,
                true,
                Creators,
                TensorOperations.Expand,
                Argument
                );
            }

            return new Tensor(m);
        }

        public static Tensor Neg(Tensor A)
        {
            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.data * -1.0f,
                                  true,
                                  Creators,
                                  TensorOperations.Negation);
            }

            return new Tensor(A.data * -1.0f);
        }

        public static Tensor Add(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(A.data + B.data,
                                  true,
                                  Creators,
                                  TensorOperations.Addition);
            }

            return new Tensor(A.data + B.data);
        }

        public static Tensor Sub(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(A.data - B.data,
                                  true,
                                  Creators,
                                  TensorOperations.Substraction);
            }

            return new Tensor(A.data - B.data);
        }

        public static Tensor Mul(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(Matrix.DeltaMult(A.data, B.data),
                                  true,
                                  Creators,
                                  TensorOperations.Multiplication);
            }

            return new Tensor(Matrix.DeltaMult(A.data, B.data));
        }

        public static Tensor Sum(Tensor A, AxisZero dim)
        {
            if (A.AutoGrad)
            {
                var Argument = new List<object>() { dim };
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.data.Sumatory(dim),
                                  true,
                                  Creators,
                                  TensorOperations.Sumatory,
                                  Argument);
            }

            return new Tensor(A.data.Sumatory(dim));
        }

        public static Tensor Transp(Tensor A)
        {
            if (A.AutoGrad)
            {
                var Creators = new List<Tensor>() { A };
                return new Tensor(A.data.T,
                                  true,
                                  Creators,
                                  TensorOperations.Transpose);
            }

            return new Tensor(A.data.T);
        }

        public static Tensor MatMul(Tensor A, Tensor B)
        {
            if (A.AutoGrad && B.AutoGrad)
            {
                var Creators = new List<Tensor>() { A, B };
                return new Tensor(Matrix.MatMult(A.data, B.data),
                                  true,
                                  Creators,
                                  TensorOperations.MatrixMultiplication);
            }

            return new Tensor(Matrix.MatMult(A.data, B.data));
        }
    }
}