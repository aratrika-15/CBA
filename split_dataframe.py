def define_splitting(data):
    block_size = int(len(data) / 10)
    split_point = [k * block_size for k in range(0, 10)]
    split_point.append(len(data))
    return split_point