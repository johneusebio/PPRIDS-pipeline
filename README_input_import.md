# input_import

input_import is a Python package for reading in multiple datapoints, each containing a user-determined amount of keyword-defined variables. This is useful for parallelizing operations across multiple different files and configurations.

# Usage

The main function by which users will interact with is `interpret_input`, which outputs a pandas dataframe in which each row is an independent datapoint corresponding to the row of the input text file and the columns correspond to a 

```python
interpret_input(filepath="/path/to/input.txt", keywords=["keyword1", "keyword2", "keyword3"])
```

The input text file must be formatted as follows:

```
KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]
KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]
KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]
```