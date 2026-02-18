labelled_data["is_unlabelled"] = np.zeros(len(labelled_data['y_labels']))
unlabelled_data["is_unlabelled"] = np.ones(len(unlabelled_data['y_labels']))

embeddings = np.concatenate([labelled_data["embeddings"],
                             unlabelled_data["embeddings"]]) # This is ALL (l+u)

labels = np.concatenate([labelled_data["y_labels"],unlabelled_data["y_labels"]])

classes = np.concatenate([labelled_data["y_class"],
                          unlabelled_data["y_class"]])

audio_nums = np.concatenate([labelled_data["audio_num"],
                              unlabelled_data["audio_num"]])

is_unlabelled = np.concatenate([labelled_data["is_unlabelled"],
                                unlabelled_data["is_unlabelled"]])

#transformations = np.concatenate([labelled_data["transformations"],
#                                 unlabelled_data["transformations"]])

idx = np.random.permutation(len(embeddings)) # shuffling so that labelled and unlabelled data get uniformly distributed among batches

embeddings = embeddings[idx]
labels = labels[idx]
classes = classes[idx]
audio_nums = audio_nums[idx]
#transformations = transformations[idx]
is_unlabelled = is_unlabelled[idx]
