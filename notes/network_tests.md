## Good
### Trial 1
* Use GraphConv with edge weights as weights
* Use one-hot with an additional embedding for edge classes
  * Let embedding dimension be comparable to x_src, x_dst
* Multiply class embedding with weights
* Reaches ~86-88% accuracy fast

### Trial 2
* Fix "correct" distances and labels (outside/inside edge means opposite) for virtual edges
* GraphConv with edge weights as weights
* Tanh as activation
* Create edge_feat as `[x_src, w, x_dst]`
* Add GraphNorm between linear layers
* Set learning rate 0.01
* RESULT: Accuracy of 97% after 10 epochs, 10000 graphs in dataset
  * Better at predicting correct class for 0s but mostly correct for 1 as well
  * torch.manual_seed(2) works, torch.manual_seed(11) goes from high accuracy to low... 
  * Must figure out way to make backprop more stable!
    * Setting losses like below seems to help
    * Works for `torch.manual_seed(2/11/?)`
    * `torch.manual_seed(4)` collapses...
    * Changing from Adam to SGD seems to help! Gives slower training but it appears to converge
  ```
  _desired_weights = torch.zeros(weights.shape)
            if prediction == label:
                norm = max((~match_mask).sum(), 1)
                _desired_weights[~match_mask] = 1 / norm
            else:
                _desired_weights[~match_mask] = 0
``` 