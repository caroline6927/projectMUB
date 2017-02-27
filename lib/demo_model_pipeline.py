# for demo only
def get_train_test(data, split):
    train_data_length = int(len(data)*split)
    print('number of observations in training data is: %d' % train_data_length)
    print('number of observations in testing data is: %d' % (len(data)-train_data_length))
    train = data[:train_data_length]
    test = data[-(len(data)-train_data_length):]
    test.reset_index(drop=True, inplace=True)
    return train, test


def get_model_accuracy(predicted, observed):
    score = sum(predicted == observed)/len(predicted)
    false_pos = sum((predicted == 10) & (observed == -10))/len(predicted)
    false_neg = sum((predicted == -10) & (observed == 10)) / len(predicted)

    print('prediction score on test data is %f' % score)
    print('false positive rate is %f' % false_pos)
    print('false negative rate is %f' % false_neg)