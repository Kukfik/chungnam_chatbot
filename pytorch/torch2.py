import torch1
from torch1 import Torch

class PyTorch(Torch):
    def __init__(self, name):
        super().__init__(name)
        self.pyname = "2"

    def __str__(self):
        return 'pyname: '+ self.pyname + "\nname:" + self.name
    
    def __eq__(self, value):
        return self.pyname == value.pyname


def main():
    t = Torch("choi alina")
    t2 = PyTorch("ts")
    t3 = PyTorch("ts")
    t.print()
    t2.print()
    print(t2)
    # print(isinstance(t, object))
    print(t2 == t3)
    #비교연산자 __gt__ -> > __le__ -> < __ne__ ->
if __name__ == '__main__':
    main()
