##### Network improvements
* Could extend edge attribute from (weight, equivalence class) to (weight, equivalence class, edge type (x-x, z-z, z-x))
  * Use z = torch.isin(edges, valid_nodes), z.sum() and compare 0, 1, 2
