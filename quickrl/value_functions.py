import torch
from torch import nn


class ValueFunction:
    def __init__ (
        self, 
        VorQ,
        shape, 
        processing=None
    ):

        if VorQ == 'value' or VorQ in 'vV':
            self.type = 'v'
        else:
            self.type = 'q'

        self.shape = shape
        self.processing_func = processing

        self.loss_func = None
        self.optimizer = None
    
    
    def process (function):
        """This function should be able to process any form of input to either a value function or an action value function.
        WARNING: always call 'function' by specifying the 'state' and 'action' args first, and then the additional args."""

        def wrapper (self, state, *args, **kwargs):

            # CASE 1: the input has already been processed or does not need processing
            processed = kwargs.pop('processed', False)
            if self.processing_func is None or processed:

                # 1.a: 'function' is intended to be used as a value function but it must accept both the 'state' and 'action' args.
                #      in this case we pass it a 'None' in place of the 'action' argument. The 'NoneType' ecception must be handled by 'function' 
                if self.type == 'v' and 'action' in function.__code__.co_varnames:
                    return function(self, state, None, *args, **kwargs)

                # 1.b: all other cases
                #      'function' is intended to be used as a value function and it does not accept 'action' argument
                #      'function' is intended to be used as an action-value function and it does accept 'action' argument
                else:
                    return function(self, state, *args, **kwargs)
         
            
            # CASE 2: we have to process the state or state-action pair
            if self.type == 'v':
                state = self.processing_func(state)
                if 'action' in function.__code__.co_varnames:
                    return function(self, state, None, *args)
                else:
                    return function(self, state, *args)
            
            else:
                if 'action' in function.__code__.co_varnames:
                    state, action = self.processing_func(state, args[0])
                    return function(self, state, action, *args[1:])
                else:
                    pair = self.processing_func(state, args[0])
                    return function(self, pair, *args[1:])

        return wrapper
    
    
class TabularValue (ValueFunction):
    def __init__ (
        self,
        VorQ,
        state_shape, 
        action_shape, 
        processing=None
    ):
        
        ValueFunction.__init__(
            self,
            VorQ,
            state_shape + action_shape, 
            processing
        )

        if action_shape is None:
            self.table = torch.rand(*state_shape)
        else:
            self.table = torch.rand(*action_shape, *state_shape)


    @ValueFunction.process
    def __call__ (self, pair, _):
        return self.table[pair]


    def update (self, *args, **kwargs):
        pass


class SparseLinear (ValueFunction):
    def __init__ (
        self, 
        VorQ,
        tiles_number,
        processing=None
    ):

        ValueFunction.__init__(
            self, 
            VorQ,
            tiles_number, 
            processing
        )
        
        self.weights = torch.rand(tiles_number, requires_grad=True)


    def parameters (self):
        yield self.weights


    @ValueFunction.process
    def __call__ (self, active_features):
        return self.weights.gather(0, active_features).sum()
    
    
    @ValueFunction.process
    def update (self, active_features, target):
        loss = self.loss_func(target, self(active_features, processed=True)) 
        loss.backward()
        self.optimizer.step()
        self.weights.grad.zero_()
        return loss


class TorchNet (ValueFunction):
    def __init__(
        self, 
        VorQ,
        network, 
        processing=None
    ):

        for k, p in enumerate(network.parameters()):
            if not k:
                input_shape = p.shape[1]
        output_shape = p.shape[0]

        super().__init__(VorQ, (input_shape, output_shape), processing)
        self.function = network

        self.loss_func = None
        self.optimizer = None
   

    def parameters (self):
        return self.function.parameters()


    @ValueFunction.process
    def __call__ (self, state, action=None):
        if action is None:
            return self.function(state)

        axis = 0 if not action.shape or action.shape[0] <= 1 else 1 
        if axis: action = action.view(-1,1)
        return self.function(state).gather(axis, action).view(-1)
    
    
    @ValueFunction.process
    def update (self, state, action, target):
        prediction = self(state, action)
        losses = self.loss_func(target, prediction)

        losses.mean().backward()
        self.optimizer.step()
        self.function.zero_grad()
        return losses
