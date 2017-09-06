import numpy as np
from keras import backend as K


def loglik_discrete(y, u, a, b, epsilon=1e-35):
    hazard0 = K.pow((y + epsilon) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)

    loglikelihoods = u * \
        K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1
    return loglikelihoods


def loglik_discrete_jupyter(y, u, a, b, K, epsilon=1e-35):
    return loglik_discrete(K.variable(y), K.variable(u), K.variable(a), 
                           K.variable(b), epsilon=epsilon).eval(session=K.get_session())


def batch_generator(df, max_date, batch_size=32, seed=20, shuffle=True):
    uniq_users = df.user_uid.unique()
    number_of_batches = np.ceil(len(uniq_users) / batch_size).astype(int)

    np.random.seed(seed)
    idx = np.random.permutation(df.index)
    df = df.loc[idx]
    num_iter = 0

    id_col = 'user_uid'
    t_col = 't_elapsed'
    date_col = 'date_event'
    event_col = 'event'

    df[t_col] = df.groupby([id_col], group_keys=False).apply(
        lambda r: r[date_col] - r[date_col].min())

    max_seq_len = df[t_col].nunique()

    while True:
        batch_users = uniq_users[num_iter * batch_size: (num_iter + 1) * batch_size]

        df_batch = df[df['user_uid'].isin(batch_users)]
        df_batch = df_batch.sort_values(by=['user_uid', 'date_event'])
        df_batch['event'] = 1

        # dt = "wallclock time", global timestep i.e 2012-01-01,...
        abs_time_col = 'dt'

        discrete_time = True
        # Does the explicit rows in dataset cover each sequence whole life?
        sequences_terminated = False

        if not sequences_terminated:
            # Assuming each sequence has its own start and is not terminated by last event:
            # Add last time that we knew the sequence was 'alive'.
# !!!!!!            
#             df_batch['user_uid'] = 
#             df_batch['']
#             qq.groupby('user_uid', as_index=False).last()
              df_batch_ids = df_batch[['user_uid']].drop_duplicates()
              df_batch_ids[date_col] = max_date
              df_batch = pd.merge(df_batch_ids, df_batch, how='outer')
#               df_batch_last = df_batch.groupby('user_uid', as_index=False).last()
              
            
#             df_batch = tr.df_join_in_endtime(df_batch,
#                        constant_per_id_cols=[id_col], 
#                        abs_time_col=abs_time_col)
            # Warning: fills unseen timesteps with 0

    #     df_batch[t_col] = df_batch.groupby([id_col], group_keys=False).apply(
    #         lambda r: r[date_col] - r[date_col].shift(1)).shift(-1)

        df_batch[t_col] = df_batch.groupby([id_col], group_keys=False).apply(
            lambda r: r[date_col] - r[date_col].min())

        df_batch[t_col] = df_batch[t_col].dt.days

        feature_cols = sorted(set(df_batch.columns) - set(
            [id_col, t_col, event_col, date_col]))

        unique_ids = df_batch[id_col].unique()
        n_seqs = len(unique_ids)
        n_features = len(feature_cols)
        

        # clip negative values with -1
        df_batch[feature_cols] = df_batch[feature_cols].clip_lower(-1)

        x = tr.df_to_padded(df_batch,feature_cols,t_col=t_col, id_col=id_col, max_seq_len=max_seq_len)
        # x = tr.df_to_padded(df_batch,feature_cols,t_col=t_col)
        events = tr.df_to_padded(df_batch,['event'],t_col=t_col, id_col=id_col, max_seq_len=max_seq_len).squeeze() # For tte/censoring calculation 

        discrete_time = True

        n_timesteps = x.shape[1]
        n_features  = x.shape[-1]
        n_sequences = x.shape[0]
        seq_lengths = (False==np.isnan(x[:,:,0])).sum(1)

        x = np.log(x+2.) # log-kill outliers, decent method since we have positive data

        padded_t = None

        y = np.zeros([n_sequences,n_timesteps,2])

        n_timesteps_to_hide = np.floor(n_timesteps*0.1).astype(int)

        x_train      = tr.left_pad_to_right_pad(tr.right_pad_to_left_pad(x)[:,:(n_timesteps-n_timesteps_to_hide),:])
        y_train      = tr.left_pad_to_right_pad(tr.right_pad_to_left_pad(y)[:,:(n_timesteps-n_timesteps_to_hide),:])
        events_train = tr.left_pad_to_right_pad(tr.right_pad_to_left_pad(events)[:,:(n_timesteps-n_timesteps_to_hide)])

        n_train     = x_train.shape[0]
        seq_lengths_train = (False==np.isnan(x_train[:,:,0])).sum(1)

        # Calculate TTE/censoring indicators after split.
        y_train[:,:,0] = tr.padded_events_to_tte(events_train,discrete_time=discrete_time,t_elapsed=padded_t)
        y_train[:,:,1] = tr.padded_events_to_not_censored(events_train,discrete_time)

        y[:,:,0] = tr.padded_events_to_tte(events,discrete_time=discrete_time,t_elapsed=padded_t)
        y[:,:,1] = tr.padded_events_to_not_censored(events,discrete_time)

        # NORMALIZE
        x_train,means,stds = tr.normalize_padded(x_train)
        x,_,_         = tr.normalize_padded(x,means,stds)

        # HIDE the truth from the model:
        if discrete_time:
            x = tr.shift_discrete_padded_features(x)
            x_train = tr.shift_discrete_padded_features(x_train)

        x_ = x_train
        y_ = y_train

        for i in range(n_train):
            this_seq_length = seq_lengths_train[i]
            if this_seq_length>0:
                x_[i,this_seq_length:,:] = 0
                y_[i,this_seq_length:,:] = 0


        num_iter += 1
        yield x_, y_

        if (num_iter == number_of_batches):
            if shuffle:
                idx = np.random.permutation(df.index)
                df = df.loc[idx]
            num_iter = 0
