import numpy as np

def cross_validation_xy(X, y, k: int, shuffle=True, random_state=1):
    assert k > 1
    data_new = X.copy()
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(data_new)
    test_size = int(data_new.shape[0] / k)
    for i in range(k):
        if not i:
            yield {'X_train': data_new[test_size:], 'y_train': y[test_size:],
                   'X_test': data_new[:test_size], 'y_test': y[:test_size],
                   'k': i+1}
        else:
            yield {'X_train': np.vstack([data_new[0:test_size*i], data_new[test_size*(i+1):]]),
                   'y_train': np.hstack([y[0:test_size*i], y[test_size*(i+1):]]),
                   'X_test': data_new[test_size*i:test_size*(i+1)],
                   'y_test': y[test_size*i:test_size*(i+1)],
                   'k': i+1}

def k_fold_dataframe(df, k, shuffle=True, seed=None, frac=1):
    assert k > 1
    if shuffle:
        if seed:
            df_new = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
        else:
            df_new = df.sample(frac=frac).reset_index(drop=True)
    else:
        df_new = df.copy()
    test_size = df_new.shape[0] // k
    for i in range(k):
        if not i:
            yield {'train': df_new.iloc[test_size:], 'test': df_new.iloc[:test_size], 'k': i+1}
        else:
            yield {'train': pd.concat([df_new.iloc[0:test_size*i], df_new.iloc[test_size*(i+1):]]),
                   'test': df_new.iloc[test_size*i:test_size*(i+1)], 'k': i+1}
def split_k(X, y, k: int, shuffle=True, random_state=1):
    assert k > 1
    assert X.shape[0] == y.shape[0]
    data_new = X.copy()
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(data_new)
    test_size = int(data_new.shape[0] / k)
    for i in range(k):
        yield {'X': data_new[test_size*i:test_size*(i+1)], 'y': y[test_size*i:test_size*(i+1)]}
        
def cross_validation_x(X, k: int, shuffle=True, random_state=1):
    assert k > 1
    data_new = X.copy()
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(data_new)
    test_size = int(data_new.shape[0] / k)
    for i in range(k):
        if not i:
            yield [data_new[test_size:], data_new[:test_size] ,i+1] # [train, test, k_fold]
        else:
            yield [np.concatenate([data_new[0:test_size*i], data_new[test_size*(i+1):]]),
                   data_new[test_size*i:test_size*(i+1)],
                   i+1] # [train, test, k_fold]