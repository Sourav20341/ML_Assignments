def accuracy_score(Y_actual, Y_predict):
    n = len(Y_actual)
    score = 0
    for i in range(n):
        if Y_actual[i] == Y_predict[i]:
            score += 1
    return score
