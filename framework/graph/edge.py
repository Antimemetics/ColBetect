class Edge:

    def __init__(self, uid1, uid2, weight):
        self.uid1 = uid1
        self.uid2 = uid2
        self.weight = weight

    def get_edge_dict(self):
        edge_dict = {'uid1': self.uid1,
                     'uid2': self.uid2,
                     'weight': self.weight}
        return edge_dict