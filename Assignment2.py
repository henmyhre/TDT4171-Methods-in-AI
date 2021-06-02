import numpy as np

# start prior probs in vector
f_start = np.array([0.5, 0.5])

# Matrix for prob for fish given prior fish
F = np.array([
  [0.8, 0.3],
  [0.2, 0.7]
])

# Matrix for prob for fish given birds present
B = np.array([
  [0.75, 0.2],
  [0.25, 0.8]
])

# 0 == true and 1 == false
evidence = [0, 0, 1, 0, 1, 0]

# adding 2's to differ form 0 and 1 up to length 30 for task 1c. See forward algorithm ans its loop to understand why
extended_evidence = evidence + 24*[2]


# normalizes a vector
def normalize(v):
  # Some fancy stuff from the world wide web
  return [float(i)/sum(v) for i in v]

def forward(evidence, k):
  f = f_start
  for e_i in evidence[0:k]:
    # diagonal matrix init. Stays this way if the evidence does not equal 0 or 1. i.e. stays this way for the last 24 evidence in 1c.
    Diag = np.array([
      [1, 0],
      [0, 1]
    ])
    # Creates a diagonal matrix with the probabilities given (not) birds
    if e_i == 0 or e_i == 1:
      Diag = np.array([
        [B[e_i][0], 0], 
        [0, B[e_i][1]]
      ])
    # Finds the probability by matrix multipliplying with Diag, F and f
    f = np.matmul(np.matmul(Diag, F), f)
  res = f
  return res

def backward(evidence, k, t):
  # init value for result
  res = np.ones(2)
  # going backward from t to k
  for i in range(t-1, k-1, -1):
    e_i = evidence[i]
    # diagonal matrix with the fish given (not) birds values
    Diag = np.array([
      [B[e_i][0], 0], 
      [0, B[e_i][1]]
    ])
    res = np.matmul(np.matmul(F.transpose(), Diag), res)
  return res

# Task 1b:
print("\n")
print("\n")
print("1b")

# Looping through all the evidence 
for i in range(1, len(evidence)+1):
  # Finding the forward probability and normalizing 
  prob = forward(evidence, i)
  prob = normalize(prob)
  # Printing depends on whether there are birds present or not
  if evidence[i-1]:
    print("Day", i, "(not birds) given day", i-1)
      
  else:
    print("Day", i, "(birds) given day", i-1)
  print("Probability:", prob)

# Task 1c:
print("\n")
print("\n")
print("1c")


# Looping through the extended evidence
for i in range(7, len(extended_evidence)):
  # FInding the forward probability and normalizing 
  prob = forward(extended_evidence, i)
  prob = normalize(prob)
  
  print("Day", i, "given the first 6 results and evidence")
  print("Probability:", prob[0])


# Task 1d:
print("\n")
print("\n")
print("1d")

for i in range(6):
  # Finding the forward probability
  prob_forward = forward(evidence, i)
  # Finding the backward probability
  prob_backward = backward(evidence, i, 6)
  # The total probability is those two multiplied
  prob = prob_forward*prob_backward
  # Normaliing the probability
  prob = normalize(prob)
  print("Day", i, "given the results from both forward and backward")
  print("Probability:", prob[0])

# Task 1e:
print("\n")
print("\n")
print("1e")

# Path probabilities init
Probs = np.ones((2, 6))

# Probability for first instance
Probs[:,0] = B[evidence[0]]*(np.matmul(F, f_start))

# Normalizing the first column
Probs[:,0] = normalize(Probs[:,0])

for i in range(1, 6):
  # Finding the most likely path
  Probs[:,i] = np.max(Probs[:,None,i-1]*F.transpose())*B[evidence[i]]
  # Normalizing the i-column
  Probs[:,i] = normalize(Probs[:,i])

# The path is the combination of zeros and ones that contains the highest probabilities 
path = np.argmax(Probs, axis=0)

print("The most probable path:")
print(path)

print("0=true and 1=false")
