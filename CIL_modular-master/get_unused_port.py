
if __name__ == "__main__":
    path="/scratch/yang/aws_data/CIL_modular_data/port.txt"

    # assuming no other process is operating on port.txt
    with open(path, "r") as f:
        port = int(f.readline())

    port = port + 3

    with open(path, "w") as f:
        f.write(str(port))

    print(port)