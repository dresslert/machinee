class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        # Inicializa a regressao linear, com uma taxa de aprendizado e numeros de epocas
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Inicializa pesos e tendencias como None(sera atribuido durante o treinamento)
        self.weights = None
        self.bias = None

    def train(self, x, y):
        # inicializa pesos e polarização aleatoriamente ou com zeros
        self.weights = [0.0 for _ in range(len(x[0]))]
        self.bias = 0.0

        # treina o modelo para um numero especifico de epocas
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # faz a previsao
                y_pred = self.predict(x[i])

                # calcula o erro
                error = y_pred - y[i]

                # atualiza pesos e tendencias usando gradiente descendente
                self.bias = self.bias - self.learning_rate * error
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] - self.learning_rate * error * x[i][j]
    
    def predict(self, input_data):
        # calcula a previsao usando presos e tendencias 
        prediction = self.bias
        for i in range(len(input_data)):
            prediction += self.weights[i] * input_data[i]

        return prediction
    
# exemplo de uso
# Suponha que você tenha dados de entrada X e saídas correspondentes y
X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
y = [3.0, 4.0, 5.0]

# Cria e treina o modelo
model = SimpleLinearRegression(learning_rate=0.01, epochs=1000)
model.train(X, y)

# Faz uma previsão
new_input = [4.0, 5.0]
prediction = model.predict(new_input)
print("Prediction:", prediction)