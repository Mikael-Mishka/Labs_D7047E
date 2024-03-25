import torch
import numpy
import matplotlib
import sklearn

def main():
    print("Hello World!")
    print(torch.__version__,
          numpy.__version__,
          matplotlib.__version__,
          sklearn.__version__)

if __name__ == "__main__":
    main()
    a = torch.Tensor([3])
    b = torch.Tensor([3])

    print(a*b)