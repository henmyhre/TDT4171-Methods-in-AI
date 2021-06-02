import pandas as pd
# ignoring irrelevant warning for my use-case
pd.options.mode.chained_assignment = None
import math
import numpy
import pprint

eps = numpy.finfo(float).eps

train_data = pd.read_csv (r'./train.csv')
test_data = pd.read_csv (r'./test.csv')

outcome_attribute = "Survived"


def remove_elem_from_list(l, e):
  """Removes element from list and returns list"""
  l.remove(e)
  return l


def B(q):
  """Calculates the entropy"""
  return -(q*math.log(q + eps, 2) + (1-q)*math.log((1-q) + eps, 2))


def p_n_count(examples, value):
  """Counts the positive or negative values for the goal attribute"""
  return len(examples[examples[outcome_attribute] == value].index)


def get_p_k_n_k_values(examples, value, attribute):
  """Returns p_k and n_k (the number of positive and negative outcomes) where attribute == value"""
  examples_with_value = examples[examples[attribute] == value]
  n_k = p_n_count(examples_with_value, 0)
  p_k = p_n_count(examples_with_value, 1)
  return n_k, p_k


def get_attribute_values(examples, attribute):
  """Returns all posible values for an discrete attribute"""
  return list(set(examples[attribute].tolist()))


def calculate_remainder(values, p, n):
  """Calculates the remainder using the outcomes for all possible values"""
  remainder = 0
  for values in values:
    if values[0] and values[1]:
      remainder += ((values[1] + values[0]) / (p + n)) * B(values[1] / (values[1] + values[0]))
  return remainder


def importance(attribute, examples, split_value=None):
  """Calculates the information gain for an attribute"""
  n = p_n_count(examples, 0)
  p = p_n_count(examples, 1)
  p_k_n_k_values = []
  # if attribute has split value
  if split_value != None:
    examples_greater_than_value = examples[examples[attribute] > split_value]
    examples_smaller_than_value = examples[examples[attribute] <= split_value]
    # saves the outcomes for every possible value for attribute
    for examples in [examples_greater_than_value, examples_smaller_than_value]:
      n_k = p_n_count(examples, 0)
      p_k = p_n_count(examples, 1)
      p_k_n_k_values.append([n_k, p_k])
  else:
    att_values = get_attribute_values(examples, attribute)
    # saves the outcomes for every possible value for attribute
    for value in att_values:
      n_k, p_k = get_p_k_n_k_values(examples, value, attribute)
      p_k_n_k_values.append([n_k, p_k])
  remainder = calculate_remainder(p_k_n_k_values, p, n)
  # calculates gain
  gain = B(p/(p+n)) - remainder
  return gain


def plurality_value(examples):
  """Counts n and p to return the most frequent"""
  n = p_n_count(examples, 0)
  p = p_n_count(examples, 1)
  # Returns p if equal
  return int(p >= n)


def decision_tree_learning(examples, attributes, par_examples = None, split_values = {}):
  """Creates a tree as a dictionary based on the provided attributes"""
  # returns outcome if only that outcome is possible
  for val in [0, 1]:
    if len(examples) == p_n_count(examples, val): return (val) 
  # returns most probable outcome when no more attributes
  if (len(attributes) == 0): return plurality_value(examples)
  # saves the most important attribute, i.e., the one with the most information gain
  important_attribute = attributes[0]
  max_importance = -1
  best_split = 0
  for attribute in attributes:
    # Checks every split value and saved the best if attribute has split values
    if attribute in split_values:
      possible_values = split_values[attribute]
      for i in range(0, len(possible_values) - 1):
        current_attribute_gain = importance(attribute, examples, possible_values[i])
        if current_attribute_gain > max_importance:
          max_importance = current_attribute_gain
          important_attribute = attribute
          best_split = possible_values[i]
    # Only saves max information gain and the related attribute if attribute does not have split values
    else:
      current_attribute_gain = importance(attribute, examples)
      if current_attribute_gain > max_importance:
        max_importance = current_attribute_gain
        important_attribute = attribute
  tree = {}
  tree[important_attribute] = {}
  new_attributes = remove_elem_from_list(attributes.copy(), important_attribute)
  # recursively calls decision_tree_learning with updated attributes and examples for each possible value
  if important_attribute in split_values: # Splits example in above split value and below if split value
    new_examples_up = examples[examples[important_attribute] > best_split]
    new_examples_down = examples[examples[important_attribute] <= best_split]
    tree[important_attribute]["> {}".format(best_split)] = decision_tree_learning(new_examples_up, new_attributes, examples, split_values)
    tree[important_attribute]["<= {}".format(best_split)] = decision_tree_learning(new_examples_down, new_attributes, examples, split_values)
  else: # iterates through all values if not split value
    att_values = get_attribute_values(examples, important_attribute)
    for value in att_values:
      new_examples = examples[examples[important_attribute] == value]
      tree[important_attribute][value] = decision_tree_learning(new_examples, new_attributes, examples, split_values)
  return tree
      

def find_expected_value(example, tree):
  """Finds the expected value according to the tree for a given test example"""
  # returns value when the end is reached
  if tree == 0 or tree == 1:
    return tree
  attribute = list(tree.keys())[0]
  value = example[attribute]
  new_tree = {}
  for val in list(tree[attribute].keys()):
    # Checks if the val is custom made and accesses the correct bucket
    val_list = str(val).split(" ")
    if val_list[0] == "<=" and float(value) <= float(val_list[1]):
      new_tree = tree[attribute][val]
    elif val_list[0] == ">" and float(value) > float(val_list[1]):
      new_tree = tree[attribute][val]
  # This is if val is not custom made split. Activates for all vals in 1a
  if new_tree == {}:
    new_tree = tree[attribute][value]
  return find_expected_value(example, new_tree)


def test(examples, tree):
  """Tests the tree with test data"""
  correct = 0
  false = 0
  # iterating though all rows of examples and checks if the tree produces the correct survival-output
  for index, example in examples.iterrows():
    expected_value = find_expected_value(example, tree)
    actual_value = example[outcome_attribute]
    if actual_value == expected_value:
      correct += 1
    else:
      false += 1
  return correct/(correct+false)


def find_attribute_split_values(examples, attributes, num):
  """Calculates split values for all continuous attributes"""
  all_split_values = {}
  for attribute in attributes:
    # Checks if continuous
    if attribute not in attributes_a:
      num_of_splits = num
      max_value = float(examples[attribute].max())
      min_value = float(examples[attribute].min())
      diff = max_value - min_value
      if diff < num_of_splits:
        # int(diff) + 1 is probably enough splits to be between each value. At least for integer-values
        num_of_splits = int(diff) + 1
      bucket_size = (diff) / (num_of_splits)
      # Creates a list with all possible split values
      all_split_values[attribute] = [round(min_value + bucket_size * i, 2) for i in range(num_of_splits)]
  return all_split_values


def task_1_a(train_data, test_data, attributes):
  """1a: making tree, testing and printing"""
  print("Task 1a")
  tree = decision_tree_learning(train_data, attributes_a)
  pprint.pprint(tree)
  accuracy = test(test_data, tree)
  print("Accuracy:", accuracy)


def task_1_b(train_data, test_data, attributes_with_outcome, attributes):
  """1b: making tree, testing and printing"""
  print("Task 1b")
  number_of_splits = 100000 # Write a really big number to maximize number of splits. There is a split cap based on the number of unique values for each attribute.
  split_values = find_attribute_split_values(train_data, attributes, number_of_splits)
  result_b = decision_tree_learning(train_data, attributes, None, split_values)
  pprint.pprint(result_b)
  accuracy = test(test_data[attributes_with_outcome], result_b)
  print("Accuracy:", accuracy)


def main():
  print("\n")
  print("-------------------------")
  print("\n")
  task_1_a(train_data_a, test_data, attributes_a)
  print("\n")
  print("-------------------------")
  print("\n")
  task_1_b(train_data_b, test_data, attributes_b_with_survival, attributes_b)


# attributes and data for 1a
attributes_a_with_survival = [outcome_attribute, "Sex", "Pclass"]
attributes_a = remove_elem_from_list(attributes_a_with_survival.copy(), outcome_attribute)

train_data_a = train_data[attributes_a_with_survival]

# attributes and data for 1b
attributes_b_with_survival = [outcome_attribute, "Sex", "Pclass", "Parch", "SibSp", "Fare"]
attributes_b = remove_elem_from_list(attributes_b_with_survival.copy(), outcome_attribute)

train_data_b = train_data[attributes_b_with_survival]

main()
