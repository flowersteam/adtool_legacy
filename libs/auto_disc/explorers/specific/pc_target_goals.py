import glob
import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_off
from torch_geometric.datasets import ModelNet, FakeDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

def get_random_pc(n_nodes, n_spatial_dims):
    # dataset = FakeDataset(num_graphs=1, avg_num_nodes=20, avg_degree=0, num_channels=8, edge_dim=0,
    #                       num_classes=0, task='auto', is_undirected=True, transform= None, pre_transform= None)
    # return dataset[0]
    x = torch.rand((n_nodes, n_spatial_dims)) * 2. - 1. #uniform random between [-1, 1]
    return x

class MyModelNet(ModelNet):

    catagory_to_index = {
        'airplane': 0,
        'plant': 27,
    }

    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None, categories="all", process=False, download=False):
        self.categories = categories
        if process:
            self.process = super().process
        if download:
            self.download = super().download

        super().__init__(root, name=name, train=train, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

    def process_set(self, dataset):
        if self.categories == "all":
            categories = glob.glob(osp.join(self.raw_dir, '*', ''))
            categories = sorted([x.split(os.sep)[-2] for x in categories])
        else:
            categories = self.categories

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f'{folder}/{category}_*.off')
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

def get_modelnet40_pc(data_index=0, n_nodes=1092, seed=None,
                      categories=['plant'], root='/home/mayalen/data/pc_datasets/ModelNet40'):

    dataset = MyModelNet(root=root, name='40', train=False, transform=T.NormalizeScale(),
                       pre_transform=None, pre_filter=None, categories=categories, process=False, download=False)

    data = dataset[data_index].pos
    if seed is not None:
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
    else:
        g_cpu=None
    sampled_ids,_ = torch.randperm(len(data), generator=g_cpu)[:n_nodes].sort()
    return data[sampled_ids]

def visualize(pc_pos, title='target'):
    import matplotlib.pyplot as plt

    spatial_dims = pc_pos.shape[1]
    projection = "3d" if spatial_dims==3 else "2d"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)

    plt.title(title)
    ax.scatter(*pc_pos.t(), s=2, marker='.')
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

if __name__=='__main__':
    for i in [27,1,12,8,13,22,24,38,44,58]:
        pc_target = get_modelnet40_pc(data_index=i)
        visualize(pc_target, title=f"{i}")

