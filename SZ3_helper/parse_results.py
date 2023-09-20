import ast

def parse_sz3_results(msg, filename):
    lines = msg.split("\n")
    for line in lines:
        if line.startswith("sample"):
            sample_ratio = float(line.split(":")[1].strip())
        if line.startswith("r2"):
            r2 = float(line.split(":")[1].strip())

    return sample_ratio, r2

def parse_unpred_data(msg):
    lines = msg.split("\n")
    unpred_idx = []
    for i in range(len(lines)):
        if lines[i].startswith("unpred"):
            unpred = float(lines[i].split(" ")[1].strip())
            unpred_idx.append(ast.literal_eval(lines[i-1]))

    return unpred_idx

def get_global_index(indices, dims):
    strides = [1]
    for d in dims[::-1]:
        strides.append(strides[-1] * d)
    offset = sum(i * s for s, i in zip(strides[:-1][::-1], indices))
    return offset

def get_data_by_indices(data, indices, dims):
    offset = get_global_index(indices, dims)
    return data.loc[offset]

if __name__ == "__main__":
    offset = get_global_index([2, 1, 8], [6, 8 ,11])
    print(offset)