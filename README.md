# Training Autograd System (TAS)
<b>TAS</b> is a minimalist pytorch-like machine learning framework made with Csharp. <br>
This proyect was made using as reference the work of [@iamtrask](https://twitter.com/iamtrask) and his book [Grokking Deep Learning](https://github.com/iamtrask/grokking-deep-learning)

## TODO
* More examples and testing
* Support for real tensors not just matrices
* Performance adjustments 
* More layers, Loss functions and activation functions

## How it works
<b>TAS</b> is a automatic diferentiation framework inspired by pytorch, <b>TAS</b> uses a dynamic computational graph system that allows changes in runtime which makes it perfect for experimentation at expenses of performance.
The core of <b>TAS</b> are the Tensors, a generalization of the concept of matrix for superior dimensions, sadly <b>TAS</b> only allows Tensors up to 2 dimensions which doesn't allow its use to image analisis. 
However <b>TAS</b> can me used without problem in text analisys problems, reinforcement learning and others algorithms.

## Features
<b>TAS</b> contains tools to make the machine learning process more easier.

### Loss
* Mean Squared Error

### layers
* Linear
* Sigmoid

## How to use it
<b>TAS</b> Works with tensors, but tensors works with matrices, so to start using <b>TAS</b> you need to import Matrix class and Tensor class.

```csharp
using DLFramework;
using LinearAlgebra;
```

to create a Tensor you need to do the following.

```csharp
var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
```

The first argument of the constructor is the Matrix that will be converted into a tensor, the second argument is a boolean indicating whether the tensor is marked as autograd or not, this allows the gradient flows throught the tensor. <br>

You can do math operations with the tensors like this:

```csharp
var multiplication = Tensor.MatMul (data, weights);
```

If the operands are marked as autograd the result of the operation will be autograd automatically. <br>

You can backpropagate a gradient throught the graph using the command Backward.

```csharp
loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
```            

That will help you with the gradients calculations, now you can gradient descend.

```csharp
weight.Data -= (weight.Gradient.Data * 0.1f);
weight.Gradient.Data *= 0f;
```

### Helpers
You also can use Helpers to accelerate the process.

#### SGD
The optimitation process can be improved by using StochasticGradientDescent, this class will keep all the parameters of the graph and will upldate them using the Step method.

```csharp
var weights = new List<Tensor> ();
weights.Add (new Tensor (Matrix.Random (2, 3, r), true));
weights.Add (new Tensor (Matrix.Random (3, 1, r), true));
// Instantiate and initialization
var sgd = new StochasticGradientDescent (weights, 0.1f);

for (var i = 0; i < 10; i++) {
...
// Update weights
    sgd.Step ();
...
}
```

#### Linear
A linear is a layer abstraction of an specific matrix operation, this simplifies the process of creating a neural network.

<img src="https://latex.codecogs.com/gif.latex?Y&space;=&space;aX&space;&plus;&space;B" title="Y = aX + B" />

Linear is usually use inside a sequential layer, just like this.

```csharp
var seq = new Sequential ();
seq.Layers.Add (new Linear (2, 3, r));
seq.Layers.Add (new Linear (3, 1, r));

...

for (var i = 0; i < 10; i++) {
    var pred = seq.Forward (data);
}
```

#### SigmoidLayer
This is an Activation function, used for building neural networks, just use it inside a Sequential layer.

```csharp
var seq = new Sequential ();
seq.Layers.Add (new Linear (2, 3, r));
seq.Layers.Add (new SigmoidLayer ());
seq.Layers.Add (new Linear (3, 1, r));
seq.Layers.Add (new SigmoidLayer ());
```

#### MeanSquaredError
A loss function is what we want to reduce in our problem, in this case (MSE) the function looks like this.

<img src="https://latex.codecogs.com/gif.latex?mse&space;=&space;\sum&space;diff^{2}" title="mse = \sum diff^{2}" />

You can use it like this:

```csharp
var mse = new MeanSquaredError ();
...
for (var i = 0; i < 300; i++) {
    var pred = seq.Forward (data);
    var loss = mse.Forward (pred, target);
    loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
    ...
}
```

Usually loss is not inside the Sequential layer because it's not part of the model, it will be a problem when we want to make a prediction.


## Examples
This is a Simple neural network example without <b>TAS</b> from [Simple vectorized mono layer perceptron](https://github.com/HectorPulido/Simple-vectorized-mono-layer-perceptron)

```csharp
//Parameters
int inputCount = 2;
int hiddenCount = 5;
int outputCount = 4;
int examplesCount = 4;
double learningRate = 0.1;
Random r = new Random();

//Training data           //INPUT
Matrix x = new double[,] { { 0, 0 }, 
                            { 0, 1 }, 
                            { 1, 0 }, 
                            { 1, 1 }};  
                        //DESIRED OUTPUT
                        //XNOR AND OR XOR
Matrix y = new double[,] { { 1, 0, 0, 0 }, 
                            { 0, 0, 1, 1 }, 
                            { 0, 1, 1, 1 }, 
                            { 1, 1, 1, 0 }}; 

//Weight init
Matrix w1 = (Matrix.Random(inputCount + 1, hiddenCount , r) - 0.5) * 2.0;
Matrix w2 = (Matrix.Random(hiddenCount + 1, outputCount, r) - 0.5) * 2.0;

for (int l = 0; l < 5001; l++) //epoch
{
    //Forward pass
    Matrix z1 = x.AddColumn(Matrix.Ones(examplesCount, 1));
    Matrix a1 = z1; //(Examples, input + 1)
    Matrix z2 = (a1 * w1).AddColumn(Matrix.Ones(examplesCount, 1));
    Matrix a2 = sigmoid(z2); //(examples, hidden neurons + 1) // APPLY NON LINEAR
    Matrix z3 = a2 * w2;
    Matrix a3 = sigmoid(z3); //(examples, output)                  

    //Bacpropagation
    Matrix a3Error = a3 - y; //(examples, output) //LOSS 
    Matrix Delta3 = a3Error * sigmoid(z3, true);

    Matrix a2Error = Delta3 * w2.T;
    Matrix Delta2 = a2Error * sigmoid(z2, true);
    Delta2 = Delta2.Slice(0, 1, Delta2.x, Delta2.y); //Slicing Extra delta (from biass neuron)

    w2 -= (a2.T * Delta3) * learningRate;
    w1 -= (a1.T * Delta2) * learningRate;

    double loss = a3Error.abs.average * examplesCount;
    Console.WriteLine ($"Epoch: {l} Loss: {loss}");
}           
```

This code is simplify a lot where we use <b>TAS</b>

```csharp
var r = new Random ();
var data = new Tensor ((Matrix) new double[, ] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, true);
var target = new Tensor ((Matrix) new double[, ] { { 1 }, { 0 }, { 0 }, { 1 } }, true);

var seq = new Sequential ();
seq.Layers.Add (new Linear (2, 5, r));
seq.Layers.Add (new SigmoidLayer ());
seq.Layers.Add (new Linear (5, 1, r));
seq.Layers.Add (new SigmoidLayer ());

var sgd = new StochasticGradientDescent (seq.Parameters, 1f);

var mse = new MeanSquaredError ();

for (var i = 0; i < 300; i++) {
    var pred = seq.Forward (data);
    var loss = mse.Forward (pred, target);
    loss.Backward (new Tensor (Matrix.Ones (loss.Data.X, loss.Data.Y)));
    sgd.Step ();
    Console.WriteLine ($"Epoch: {i} Loss: {loss}");
}
```
