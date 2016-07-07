"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np


# Data in format [user_id, item_id, counts]
data = np.genfromtxt ('lastfm_40000_300.csv', delimiter=",")
print data.shape

def split_to_train_val_and_test(uid, u_locs, portions):
    """
    Simple split to train, validation and test datasets. It assumes nothing about the data, and just splits it according
    to the order it's in (so most of the time I should call it with ordered data).

     INPUT:
    -------
        1. uid:         <scalar>            user identifier. For debug prints.
        2. u_locs:      <(N, ) ndarray>     user observed locations. Each row is [location_id]
        3. portions:    <(2, ) ndarray>     How much is train, how much is validation, the rest is test.

     OUTPUT:
    --------
        1. u_train, u_val, u_test:      <(N', 3) ndarray>   coo_matrix style data. [uid, lid, counts].
    """
    N = u_locs.shape[0]
    tr_idx = int(np.ceil(N * portions[0]))
    val_idx = int(np.ceil(N * (portions[0] + portions[1])))

    u_tr_locs, u_tr_counts = np.unique(u_locs[:tr_idx], return_counts=True)
    u_val_locs, u_val_counts = np.unique(u_locs[tr_idx:val_idx], return_counts=True)
    u_te_locs, u_te_counts = np.unique(u_locs[val_idx:], return_counts=True)

    if np.sum(u_tr_counts) == 0 or np.sum(u_val_counts) == 0 or np.sum(u_te_counts) == 0:
        raise AssertionError('Bad split for user %d' % uid)

    u_train = np.vstack([uid * np.ones(u_tr_locs.shape[0]), u_tr_locs, u_tr_counts]).T
    u_val = np.vstack([uid * np.ones(u_val_locs.shape[0]), u_val_locs, u_val_counts]).T
    u_test = np.vstack([uid * np.ones(u_te_locs.shape[0]), u_te_locs, u_te_counts]).T

    return u_train, u_val, u_test


# The three datasets that you want at the end
train = np.zeros([0, 3])
val = np.zeros([0, 3])
test = np.zeros([0, 3])


# Now looping on all user ids, creating the dataset for them
uids = np.unique(data[:, 0])
for u in uids:
    print u
    u_data = data[np.where(data[:, 0] == u)[0], 1:]     # [location_id, counts]
    
    # This is ad-hoc way of doing it. I'm first creating from each input [location id, counts] a list
    # with the location_id multiple times (#counts). This is how my code works, so I just did that. You can
    # probably do it better, but why bother? It's a pre-processing code
    flat = np.hstack([np.tile(u_data[i, 0], u_data[i, 1]) for i in range(u_data.shape[0])])
    u_train, u_val, u_test = split_to_train_val_and_test(u, flat, [0.6, 0.2, 0.2])

    # Adding the user data to the global params
    train = np.vstack([train, u_train])
    val = np.vstack([val, u_val])
    test = np.vstack([test, u_test])

np.savetxt('train.csv', train, delimiter=',')
np.savetxt('val.csv', val, delimiter=',')
np.savetxt('test.csv', test, delimiter=',')

