import click
import pytest 
import main

def test_main_train_no_test_no_split():
    with pytest.raises(Exception) as excinfo:
        main('train', data='data.csv')
    assert "You must provide data and model" in str(excinfo.value)

def test_main_train_with_test():
    with pytest.raises(Exception) as excinfo:
        main('train', data='data.csv', model='model.pkl', test='test.csv')
    assert "You must provide data and model" in str(excinfo.value)

def test_main_train_with_split():
    with pytest.raises(Exception) as excinfo:
        main('train', data='data.csv', model='model.pkl', split=0.2)
    assert "You must provide data and model" in str(excinfo.value)

def test_main_predict():
    with pytest.raises(Exception) as excinfo:
        main('predict', model='model.pkl', data='data.csv')
    assert "You must provide data and model" in str(excinfo.value)

def test_main_invalid_command():
    with pytest.raises(Exception) as excinfo:
        main('invalid_command', data='data.csv', model='model.pkl')
    assert "Invalid command" in str(excinfo.value)
