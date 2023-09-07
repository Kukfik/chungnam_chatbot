import time 

class Torch:
    def __init__(self, name):
        self.name = "1"
    def print(self):
        print("this is torch class\n" + "name:" + self.name)

def main():
    print("Hello, world")
    torch1 = Torch("tsoy alina")
    # torch2 = Torch()
    # torch3 = Torch()
    torch1.print()
    print(torch1.name)

if __name__ == '__main__':
    main() 