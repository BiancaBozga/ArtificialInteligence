import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Plot 3D scatter plot
def plot3D(inputs, outputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract inputs
    gdp = [input[0] for input in inputs]
    freedom = [input[1] for input in inputs]

    # Plot the data
    ax.scatter(gdp, freedom, outputs, c='b', marker='o')

    # Set labels and title
    ax.set_xlabel('Economy..GDP.per.Capita.')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness.Score')
    ax.set_title('Economy..GDP.per.Capita. and Freedom vs. Happiness')

    # Show the plot
    plt.show()

def loadData(fileName, inputVariabNames, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    # Identificarea indiceilor pentru variabilele de intrare și ieșire
    inputIndices = [dataNames.index(var) for var in inputVariabNames]
    outputIndex = dataNames.index(outputVariabName)

    # Extrageți datele de intrare și ieșire din fișierul CSV
    inputs = [[float(data[i][index]) if data[i][index] != '' else np.nan for index in inputIndices] for i in
              range(len(data))]
    outputs = [float(data[i][outputIndex]) for i in range(len(data))]

    # Calcularea mediei pentru fiecare coloană de intrare
    column_means = np.nanmean(inputs, axis=0)

    # Înlocuirea valorilor lipsă cu media corespunzătoare a coloanei
    inputs = [[val if not np.isnan(val) else column_means[index] for index, val in enumerate(row)] for row in inputs]

    return inputs, outputs
def plotDataHistogram(data, title):
        plt.hist(data, bins=20)
        plt.title(title)
        plt.show()
def plotModel_one_feature(trainInputs, trainOutputs, testInputs, testOutputs, w0, w1):
    plt.scatter([input[0] for input in trainInputs], trainOutputs, color='blue', label='Train Data')
    plt.scatter([input[0] for input in testInputs], testOutputs, color='green', marker='^', label='Test Data')
    plt.xlabel('Family')
    plt.ylabel('Happiness.Score')
    plt.title('Family vs. Happiness')
    xref = np.linspace(min([input[0] for input in trainInputs + testInputs]),
                       max([input[0] for input in trainInputs + testInputs]), 100)
    yref = w0 + w1 * xref

    plt.plot(xref, yref, 'r-', label='Learnt Model')
    plt.legend()
    plt.show()
def plotModel(trainInputs, trainOutputs, testInputs, testOutputs, w0, w1, w2):
        plt.scatter([input[0] for input in trainInputs], trainOutputs, color='blue', label='Train Data')
        plt.scatter([input[0] for input in testInputs], testOutputs, color='green', marker='^', label='Test Data')
        plt.xlabel('Economy..GDP.per.Capita.')
        plt.ylabel('Happiness.Score')
        plt.title('Economy..GDP.per.Capita. vs. Happiness')
        xref = np.linspace(min([input[0] for input in trainInputs + testInputs]),
                           max([input[0] for input in trainInputs + testInputs]), 100)
        yref = w0 + w1 * xref + w2 * np.mean([input[1] for input in trainInputs + testInputs])

        plt.plot(xref, yref, 'r-', label='Learnt Model')
        plt.legend()
        plt.show()
def ex1():
    # Directorul curent
    crtDir = os.getcwd()
    # Calea către fișier
    filePath = os.path.join(crtDir, 'data', 'v3_world-happiness-report-2017.csv')

    # Numele caracteristicilor de intrare și de ieșire
    inputVars = ['Economy..GDP.per.Capita.',
                 'Freedom']
    outputVar = 'Happiness.Score'

    # Încărcați datele
    inputs, outputs = loadData(filePath, inputVars, outputVar)

    # Afișați primele câteva date
    print('inputs:', inputs[:5])
    print('outputs:', outputs[:5])

    # 1. Plotați histogramele pentru fiecare caracteristică


    plotDataHistogram([input[0] for input in inputs], 'Economy..GDP.per.Capita.')
    plotDataHistogram([input[1] for input in inputs], 'Freedom')
    plotDataHistogram(outputs, 'Happiness Score Histogram')

    # 2. Verificați liniaritatea relației
    plt.scatter([input[0] for input in inputs], outputs)
    plt.xlabel('Economy..GDP.per.Capita.')
    plt.ylabel('Happiness.Score')
    plt.title('Economy..GDP.per.Capita. vs. Happiness')
    plt.show()

    plt.scatter([input[1] for input in inputs], outputs)
    plt.xlabel('Freedom')
    plt.ylabel('Happiness.Score')
    plt.title('Freedom vs. Happiness')
    plt.show()

    # Plot the 3D scatter plot
    plot3D(inputs, outputs)

    # split data into training data (80%) and testing data (20%)
    np.random.seed(5)
    indexes = np.arange(len(inputs))
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = np.setdiff1d(indexes, trainSample)
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # training step
    regressor = linear_model.LinearRegression()
    regressor.fit(trainInputs, trainOutputs)
    w0 = regressor.intercept_
    w1, w2 = regressor.coef_
    print('The learnt model: f(x) = ', w2, ' * x2 + ', w1, ' * x1 + ', w0)

    # Plot the model


    plotModel(trainInputs, trainOutputs, testInputs, testOutputs, w0, w1, w2)

    # Make predictions for test data
    computedTestOutputs = regressor.predict(testInputs)

    # Plot True vs Predicted values for Test Data
    plt.scatter(testOutputs, computedTestOutputs, color='green', marker='^')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted values for Test Data')
    plt.plot(testOutputs, testOutputs, 'r-')
    plt.show()

    # Compute the differences between the predictions and real outputs
    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("Prediction Error (MSE) for Test Data:", error)
def ex1_with_family_only():
    # Directorul curent
    crtDir = os.getcwd()
    # Calea către fișier
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')

    # Numele caracteristicii de intrare și de ieșire
    inputVar = 'Family'
    outputVar = 'Happiness.Score'

    # Încărcați datele
    inputs, outputs = loadData(filePath, [inputVar], outputVar)

    # Afișați primele câteva date
    print('inputs:', inputs[:5])
    print('outputs:', outputs[:5])

    # Plotați histograma caracteristicii
    plotDataHistogram(inputs, inputVar)

    # Verificați liniaritatea relației
    plt.scatter(inputs, outputs)
    plt.xlabel(inputVar)
    plt.ylabel('Happiness.Score')
    plt.title(inputVar + ' vs. Happiness')
    plt.show()

    # split data into training data (80%) and testing data (20%)
    np.random.seed(5)
    indexes = np.arange(len(inputs))
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = np.setdiff1d(indexes, trainSample)
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # training step
    regressor = linear_model.LinearRegression()
    regressor.fit(trainInputs, trainOutputs)
    w0 = regressor.intercept_
    w1 = regressor.coef_[0]

    # Plot the model
    plotModel_one_feature(trainInputs, trainOutputs, testInputs, testOutputs, w0, w1)

    # Make predictions for test data
    computedTestOutputs = regressor.predict(testInputs)

    # Plot True vs Predicted values for Test Data
    plt.scatter(testOutputs, computedTestOutputs, color='green', marker='^')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted values for Test Data')
    plt.plot(testOutputs, testOutputs, 'r-')
    plt.show()

    # Compute the differences between the predictions and real outputs
    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("Prediction Error (MSE) for Test Data:", error)




def multiplication(first_matrix, second_matrix):

    rows_first = len(first_matrix)
    cols_second = len(second_matrix[0])
    cols_first = len(first_matrix[0])


    result = [[0] * cols_second for _ in range(rows_first)]


    for i in range(rows_first):
        for j in range(cols_second):
            result[i][j] = sum(first_matrix[i][k] * second_matrix[k][j] for k in range(cols_first))

    return result

def get_minor_signed_determinant_3d_matrix(matrix: list, nodes: [int, int]):
    #sarus
        return matrix[(nodes[0] + 1) % 3][(nodes[1] + 1) % 3] * matrix[(nodes[0] + 2) % 3][(nodes[1] + 2) % 3] - \
                     matrix[(nodes[0] + 1) % 3][(nodes[1] + 2) % 3] * matrix[(nodes[0] + 2) % 3][(nodes[1] + 1) % 3]

def inverted(matrix):

    det=  matrix[0][0] * get_minor_signed_determinant_3d_matrix(matrix, [0, 0]) + \
            matrix[0][1] * get_minor_signed_determinant_3d_matrix(matrix, [0, 1]) + \
            matrix[0][2] * get_minor_signed_determinant_3d_matrix(matrix, [0, 2])
    transpose = [[el1, el2, el3] for el1, el2, el3 in zip(matrix[0], matrix[1], matrix[2])]
    inverted_matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            M = []
            for ii in range(3):
                r = []
                for jj in range(3):
                    if ii != i and jj != j:
                        r.append(transpose[ii][jj])
                if r:
                    M.append(r)
            d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
            put = (-1) ** (i + j + 2)
            row.append(put * d * (1 / det))
        inverted_matrix.append(row)
    return inverted_matrix
def invert_matrix(matrix):
    # Inițializăm matricea identitate
    identity = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]


    augmented_matrix = [row + identity_row for row, identity_row in zip(matrix, identity)]


    for i in range(3):

        if augmented_matrix[i][i] == 0:

            for k in range(i + 1, 3):
                if augmented_matrix[k][i] != 0:
                    augmented_matrix[i], augmented_matrix[k] = augmented_matrix[k], augmented_matrix[i]
                    break

        pivot = augmented_matrix[i][i]
        for j in range(i, 6):
            augmented_matrix[i][j] /= pivot

        for k in range(3):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(i, 6):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]


    inverted_matrix = [row[3:] for row in augmented_matrix]
    return inverted_matrix


def my_linear_regression(gdp, freedom, outputs):
    x = [[1, el1, el2] for el1, el2 in zip(gdp, freedom)]
    ones = []
    for _ in range(len(gdp)):
        ones.append(1)
    x_transpose = [ones, gdp, freedom]

    m = multiplication(x_transpose, x)
    m_inv = invert_matrix(m)

    res = multiplication(m_inv, x_transpose)

    outputs_transpose = [[el] for el in outputs]
    final_res = multiplication(res, outputs_transpose)

    w0, w1, w2 = final_res[0][0], final_res[1][0], final_res[2][0]
    print('The learnt model: f(x) = ', w2, ' * x^2 + ', w1, ' * x + ', w0)
def display_menu():
    print("Meniu:")
    print("1. Pb 1")
    print("2. Pb 2")
    print("3. Ieșire")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Directorul curent
    crtDir = os.getcwd()
    # Calea către fișier
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')

    # Numele caracteristicilor de intrare și de ieșire
    inputVars = ['Economy..GDP.per.Capita.',
                 'Freedom']
    outputVar = 'Happiness.Score'

    # Încărcați datele
    inputs, outputs = loadData(filePath, inputVars, outputVar)
    gdpr = [row[inputVars.index('Economy..GDP.per.Capita.')] for row in inputs]
    freedom = [row[inputVars.index('Freedom')] for row in inputs]




    print('\nUsing my code:')
    my_linear_regression(gdpr, freedom, outputs)

    while True:
        display_menu()
        choice = input("Selectați o opțiune: ")

        if choice == '1':
            print("Ai selectat Opțiunea 1")
            ex1_with_family_only()
        elif choice == '2':
            print("Ai selectat Opțiunea 2")
            ex1()
        elif choice == '3':
            print("La revedere!")
            break
        else:
            print("Opțiune invalidă! Te rog să selectezi o opțiune validă.")

