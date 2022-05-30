class Node:

    def __init__(self, uid, date, user, tag):
        self.uid = uid
        self.date = date
        self.user = user
        self.tag = tag

        self.pc_list = []
        self.c_num = 0
        self.d_num = 0

    def add_line_dict(self, line_dict):
        self.pc_list.append(line_dict['pc'])
        if line_dict['act'] == 'Connect':
            self.c_num += 1
        else:
            self.d_num += 1

    def get_node_dict(self):
        node_dict = {'uid': self.uid,
                     'date': self.date,
                     'user': self.user,
                     'tag': self.tag,
                     'pc_list': self.pc_list,
                     'c_num': self.c_num,
                     'd_num': self.d_num}
        return node_dict
