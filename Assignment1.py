# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression


#%matplotlib inline

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'#change file path to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """

  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#==============================================================================
#Problem 1 Let's take a peek at some of the data to make sure it looks sensible. 
#Each exemplar should be an image of a character A through J rendered in a different font.
# Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
#==============================================================================
basedir = 'notMNIST_large'
for label in os.listdir(basedir):
	if  'pickle' in label:
		continue
	imgname = os.listdir(basedir + '/' + label)[0]
	img = Image(basedir + '/' + '/' +imgname)
	print(label)
	display(img)

#==============================================================================
#End problem 1
#==============================================================================

#LOAD DATA IN A 3D ARRAY

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

#==============================================================================
#Problem 2 Let's verify that the data still looks good. Displaying a sample of the 
#labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#==============================================================================

basedir = 'notMNIST_large'
for label in os.listdir(basedir):
	if 'pickle' not in label:
		continue
	dataset = pickle.load(open(basedir + '/' + label, "rb"))
	plt.imshow(dataset[0])
	plt.show()

#==============================================================================
#End problem 2
#==============================================================================
#==============================================================================
#Problem 3 Verify that the data is balanced
#==============================================================================

basedir = 'notMNIST_large'
for label in os.listdir(basedir):
	if 'pickle' not in label:
		continue
	dataset = pickle.load(open(basedir + '/' + label, "rb"))
	print(label)
	print('Mean', dataset.mean())
	print('Std', dataset.std())
	print('-' * 20)

#==============================================================================
# Merge and prune the training data as needed. Depending on your computer 
#setup, you might not be able to fit it all in memory, and you can tune train_size as needed. 
#The labels will be stored into a separate array of integers 0 through 9.
#==============================================================================

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


#============================================================================
#Randomize the data
#============================================================================
def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#============================================================================
#Problem 4 Check he data is still good after shuffling
#============================================================================

print('Train mean', train_dataset.mean())
print('Train std', train_dataset.std())
print('-' * 20)

print('Test mean', test_dataset.mean())
print('Test std', test_dataset.std())
print('-' * 20)

print('Valid mean', valid_dataset.mean())
print('Valid std', valid_dataset.std())
print('-' * 20)


print('train')
for i in range(5):
    plt.imshow(train_dataset[i])
    plt.show()
    
print('test')
for i in range(5):
    plt.imshow(test_dataset[i])
    plt.show()
    
print('valid')
for i in range(5):
    plt.imshow(valid_dataset[i])
    plt.show()

#============================================================================
#Problem 4 Ends then Save data for later use
#============================================================================

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

#============================================================================
#By construction, this dataset might contain a lot of overlapping samples, 
#including training data that's also contained in the validation and test set! 
#verlap between training and test can skew the results if you expect to use 
#your model in an environment where there is never an overlap, but are actually 
#ok if you expect to see training samples recur when you use it. Measure how much
#overlap there is between training, validation and test samples.

#Problem 5 
#First, define a function to detect similarity between 2 image sets. This is achieved by 
#flattening matricies into vectors and taking cosine similarities
#============================================================================

def get_similarities_for_sets(setA, setB):
    lenA = len(setA)
    lenB = len(setB)
    
    vectorsA = setA.reshape(lenA, -1)
    vectorsB = setB.reshape(lenB, -1)
    
    numerator = vectorsA.dot(vectorsB.T)
    
    normsA = np.sqrt((vectorsA * vectorsA).sum(axis=1))
    normsB = np.sqrt((vectorsB * vectorsB).sum(axis=1))
    
    denominator = np.outer(normsA, normsB)
    
    result = numerator / denominator
    
    assert result.shape[0] == lenA
    assert result.shape[1] == lenB
    
    return result

""" Because of big dimensions of data (200k rows of training data, 10k rows of validation/test data), I can't just get similarities on this machine.
But I can take into account that same images in sets have same labels! So I can detect similarities only between same labeled imaged to dramatically reduce the size of intermediate matrix"""

def get_sanitized_dataset(A, Alabels, B, Blabels, threshold=1e-4):
    Aunique = np.ones_like(Alabels)
    
    for l in np.unique(Alabels):
        idxA = Alabels == l
        idxB = Blabels == l
        
        filteredA = A[idxA]
        filteredB = B[idxB]
        
        distances = 1. - get_similarities_for_sets(filteredA, filteredB)
        Adistances = distances.min(axis=1)
        
        Aunique[idxA] = Adistances > threshold
        
    return A[Aunique == 1], Alabels[Aunique == 1]

#Lets do a trivial check: clearing of a set with itself should produce empty set!
a, b = get_sanitized_dataset(valid_dataset, valid_labels, valid_dataset, valid_labels)
print(a.shape, b.shape)

#Remove duplications with validation and training sets
Xtest_s, Ytest_s = get_sanitized_dataset(test_dataset, test_labels, valid_dataset, valid_labels)
print(Xtest_s.shape, Ytest_s.shape)

Xtest_s, Ytest_s = get_sanitized_dataset(Xtest_s, Ytest_s, train_dataset, train_labels)
print(Xtest_s.shape, Ytest_s.shape)

#sanitize validation set the same way as above
Xvalid_s, Yvalid_s = get_sanitized_dataset(valid_dataset, valid_labels, test_dataset, test_labels)
print(Xvalid_s.shape, Yvalid_s.shape)

Xvalid_s, Yvalid_s = get_sanitized_dataset(Xvalid_s, Yvalid_s, train_dataset, train_labels)
print(Xvalid_s.shape, Yvalid_s.shape)

Xtrain, Ytrain = train_dataset, train_labels
print('Train', Xtrain.shape, Ytrain.shape)
print('Valid', Xvalid_s.shape, Yvalid_s.shape)
print('Test', Xtest_s.shape, Ytest_s.shape)

#============================================================================
#Problem 6 Train a simple model on this data using 50, 100, 1000 and 5000 training 
#samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
#============================================================================

res = {}

for n in [50, 100, 1000, 5000, 10000, 50000]:
    clf = LogisticRegression(n_jobs=-1)
    clf.fit(Xtrain[:n].reshape(n, -1), Ytrain[:n])
    score = clf.score(Xtest_s.reshape(len(Xtest_s), -1), Ytest_s)
    
    res[n] = {'clf': clf, 'score': score}
    print(n, score)

keys = res.keys()
keys.sort()
plt.xscale('log')
plt.plot(keys, [res[k]['score'] for k in keys])


i = np.random.randint(len(Xtest_s))
print('Ground truth class', Ytest_s[i])
for n in keys:
    print(n, res[k]['clf'].predict(Xtest_s[i].reshape(1, -1)))
plt.imshow(Xtest_s[i])












