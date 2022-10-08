# convetr string array to nympy array
def string2nparray(string_array: str) -> np.ndarray:
    string_array_list = string_array.split()
    array = np.array([float(re.sub(r'[\[\],]', '', element)) for element in string_array_list])
    return array
