import numpy as np
import pickle
import random
import PIL.Image as im
import os
import json

# read data
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
file_path = os.path.join(file_path, 'data', 'cifar-100-python')
test_file = os.path.join(file_path, 'test')
train_file = os.path.join(file_path, 'train')
if not (os.path.exists(train_file) or os.path.exists(test_file)):
    print("Please download the cifar-100 dataset and extract it to leaf/cifar-100/data")
    exit(1)
train_data = {}
test_data = {}
with open(test_file, 'rb') as f1:
    test_data = pickle.load(f1, encoding='bytes')
with open(train_file, "rb") as f2:
    train_data = pickle.load(f2, encoding="bytes")

zip_data = list(zip(test_data[b'fine_labels'], test_data[b'coarse_labels'], test_data[b'data']))\
           + list(zip(train_data[b'fine_labels'], train_data[b'coarse_labels'], train_data[b'data']))
random.shuffle(zip_data)
fine_labels, coarse_labels, data = zip(*zip_data)

# constants and initialization
num_classes = 100  # use fine_labels by default
data_by_class = {i: [] for i in range(num_classes)}
for fine_label, data_sample in zip(fine_labels, data):
    data_by_class[fine_label].append(data_sample)

sample_per_class = 600
total_samples = 60000
num_of_users = 600  # sets 600 users by default
users = ['user' + str(i) for i in range(num_of_users)]
num_samples = [0 for i in range(num_of_users)]
user_data = {u: {'x': [], 'y': []} for u in users}

# resize to 224*224
# for label in range(num_classes):
#     for i in range(len(data_by_class[label])):
#         data_by_class[label][i] = data_by_class[label][i].reshape((32, 32, 3), order='F')
#         image = im.fromarray(data_by_class[label][i], mode='RGB')
#         image = image.resize((224, 224))
#         data_by_class[label][i] = np.asarray(image)
#         data_by_class[label][i] = data_by_class[label][i].astype(np.float) / 255
#         data_by_class[label][i] = data_by_class[label][i].reshape(224 * 224 * 3, order='F')

# distribute data for each user
# non-iid
sample_range = [i for i in range(0, sample_per_class + 1)] * 100000

for label in range(num_classes):
    num_presum = random.sample(sample_range, k=num_of_users - 1)
    num_presum.append(0)
    num_presum.append(sample_per_class)
    num_presum = sorted(num_presum)
    class_per_user = [num_presum[i] - num_presum[i - 1] for i in range(1, len(num_presum))]

    for j in range(num_of_users):
        num_samples[j] += class_per_user[j]

    temp = 0
    for j in range(num_of_users):
        for k in range(class_per_user[j]):
            user_data['user' + str(j)]['x'].append(data_by_class[label][temp])
            user_data['user' + str(j)]['y'].append(label)
        temp += class_per_user[j]

for u in users:
    zip_user_data = list(zip(user_data[u]['x'], user_data[u]['y']))
    random.shuffle(zip_user_data)
    user_data[u]['x'], user_data[u]['y'] = zip(*zip_user_data)
    user_data[u]['x'] = list(user_data[u]['x'])
    user_data[u]['y'] = list(user_data[u]['y'])

# writing to json files
MAX_USERS = 20  # max number of users per json file
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
folder_path = os.path.join(parent_path, 'data', 'all_data')
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

num_json = int(num_of_users / MAX_USERS)
if num_of_users % MAX_USERS != 0:
    num_json += 1

json_users = []
json_num_samples = []
json_user_data = {}
json_index = 0
user_count = 0
all_users = 0
for u in users:
    json_users.append(u)
    json_num_samples.append(num_samples[all_users])
    json_user_data[u] = {'x': [], 'y': []}
    len_x = len(user_data[u]['x'])
    for i in range(len_x):
        data_temp = user_data[u]['x'][i].reshape((32, 32, 3), order='F')
        image = im.fromarray(data_temp, mode='RGB')
        image = image.resize((224, 224))  # resize to 224*224
        data_temp = np.asarray(image)
        data_temp = data_temp.reshape(224 * 224 * 3, order='F')
        data_temp = data_temp.astype(np.float) / 255
        json_user_data[u]['x'].append(data_temp.tolist())
        json_user_data[u]['y'].append(user_data[u]['y'][i])

    user_count += 1
    all_users += 1
    if user_count == MAX_USERS or all_users == len(users):
        all_data = {}
        all_data['users'] = json_users
        all_data['num_samples'] = json_num_samples
        all_data['user_data'] = json_user_data

        json_path = os.path.join(folder_path, 'all_data_%d.json' % json_index)
        print('writing all_data_%d.json' % json_index)
        with open(json_path, 'w') as outfile:
            json.dump(all_data, outfile)

        user_count = 0
        json_index += 1
        json_users[:] = []
        json_num_samples[:] = []
        json_user_data.clear()
