class DimInfo:
    def __init__(self, data, param_names):
        # key: factor name, value: levels
        self.level_dict = {}
        # number of levels for each factor
        self.dimensions = []
        for term in param_names:
            print(term)
            unique_values = data[term].unique().tolist()
            print(unique_values)
            self.level_dict[term] = unique_values
            self.dimensions.append(len(unique_values))

        self.param_names = param_names

    def get_dim_len(self):
        return len(self.dimensions)

    def get_dimensions(self):
        return self.dimensions

    def get_levels_by_id(self, id):
        return self.level_dict[self.param_names[id]]

    def get_factor_names(self):
        return self.param_names