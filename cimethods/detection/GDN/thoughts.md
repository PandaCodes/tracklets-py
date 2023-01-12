# CONFLICTING neighboring CELLS:
- how to make output function smooth on coordinates (is it nesessary though?)
- how do we count? maybe predict count (n) and choose n most-probable centers instead of taking center_probabilities > 0.5

-  (maybe some <inhybiting?> connections with the surrounding cells) 
- information gradient in layers: for each pixel there is an additional output mask that defines in what direction the information vector should flow

# Another question is how do we make a good data?
- add background to the data generatior
- may use loss that enforces background to be a random noise (instead of image MSE)
