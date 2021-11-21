## Backpropagation created

0. I've just tried to implement the backpropagation algorithm
   and created the MLP classes, plus other tweaks on nn_unit
   and activation_functions.
   	- please tell me what you think!
   	- I'll go from basic (act funcs) to advanced (backprop).
1. activation_functions:
	- new activations functions added:
		- ReLu;
		- softplus;
		- gaussian;
		- SiLu;
	- I also added, for each of them, the corresponding
	  derivatives as separate functions.
2. nn_unit:
	- I removed the param weight_size; after all, we already
	  have inputs' length to determine it, so it was a bit
	  unnecessary;
	- I added the new act funcs in the same framework we already
	  used (passing the function as a string with its name);
	  plus, since such functions have additional params, in
	  order to pass them too to the unit I defined the param
	  activation in such a way to receive a list with function's
	  name first, and then its params as they appear in its
	  definition;
	- furthermore, I defined activation_prime according to the
	  given activ func;
	- I defined an update_inputs method to be called within the
	  specific layer and the MLP in general at each step of the
	  backpropagation; although I would argue why not use
	  properties and setters instead, but this we will decide then.
3. MLP (Multi-Layer Perceptron):
	contains both the layer and MLP class.
	- layer object:
		- the set of units is arranged in a dictionary, while
		nets, outputs and output_primes are stored in a
		numpy.ndarray;
		- each layer contains the corresponding key of the previous
		layer, like a "pointer";
		- an external weight matrix is present in order to
		act as a buffer during a batch before weights get updated;
		- a dropout np.ndarray shuts off certain units if needed;
		- a collective update function for inputs and weight is
		implemented.
	- MLP object:
		- length and activation function of each layer is passed
		through a tuple of int and a list of lists respectively;
		- layers are stored in a dictionary, with key values of
		increasing order going upward w.r.t. the network structure;
		- contains update function for inputs and weights.
4. Backpropagation algorithm:
	- first things first, I wrote down the MSE on the whole NN for a
	  single batch to be evaluated at each step, to check if stopping
	  condition is reached; we could also implement it in such a way
	  to stop the training if the MSE doesn't improve more than a certain
	  threshold for more than k (~50?) steps in a row;
	- a pick_batch function is implemented to randomly select l patterns
	  in the current batch;
	- I made the single step first: for each pattern in batch:
		- network units are updated;
		- picks outermost layer;
		- evaluates each Delta_w for each unit of the layer and stores it in the weight
		matrix;
		- the deltas thus obtained are taken by the next layer, that
		evaluates again its Delta_w with the hidden unit formula and so forth
		downwards;
		- once done this for every pattern in the batch, update units'
		weights with the ones stored in weight matrix;
		- the whole algorithm consists in, at each step, first evaluating current
		MSE and then the single step;
		- terminates when stopping conditions are
		reached;
		- returns streak of MSE for error evaluation.
