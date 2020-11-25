

class NeuralNetworkConfig:

    def __init__(self):
        self.maximum_rounds = 100

        """ Promising
        self.epochs = 100
        self.batch = 500
      
        self.epochs = 300
        self.batch = 2000
        """
        self.epochs = 500
        self.batch = 1000

        ### GridSearch
        self.epochs_array = [50, 100, 150]
        self.batches_array = [5, 10, 20]

        self.optimizers = ['rmsprop', 'adam']
        self.init = ['uniform', 'glorot_uniform', 'normal']

        self.optimizer = 'rmsprop'
        self.loss = 'categorical_crossentropy'
        # self.loss_function = 'sparse_categorical_crossentropy'
        # self.metrics = ['accuracy']
        self.metrics = ['categorical_accuracy']

        self.input_nodes = 25
        self.hidden_layer_1 = 512
        self.hidden_layer_2 = 960
        self.hidden_layer_3 = 512
        self.output_layer = 5
