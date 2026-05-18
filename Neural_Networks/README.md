## A Hitchhiker's Guide to Neural Networks

Slides available in this directory.

## Reading material

I simply compiled 2 sources, and I believe completely going through both would be great to build your understanding in neural networks.

- Nielsen's book [[link]](http://neuralnetworksanddeeplearning.com): Read through **at least** chapter 1 and 2

- Andrej Karpathy's backprop video: [[link]](https://www.youtube.com/watch?v=VMj-3S1tku0): How libraries like torch implement backpropagation. Basically chain rule + recursion.

- Additional resources on Object Oriented Programming (OOP) in Python:
    - https://realpython.com/python3-object-oriented-programming/
    - https://realpython.com/inheritance-composition-python/

## Assignments
1) Try to derive all 4 equations of backprop. First start with the j,k notation and then try to convert to a vector form.

2) Recreate the backprop engine in Karpathy's video, but instead of numbers each Value stores a matrix. Try to **inherit** from Numpy's ndarray class. Implement how backprop would work in case of matrix multiplication.