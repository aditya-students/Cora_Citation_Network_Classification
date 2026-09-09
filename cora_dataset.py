from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data/Cora", name="Cora")

print(dataset)
print("Number of graphs:", len(dataset))
print("Number of nodes:", dataset[0].num_nodes)
print("Number of edges:", dataset[0].num_edges)
print("Number of features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)