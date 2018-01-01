import numpy as np
import pylab as plt
import os

#######################################################################
# black box to check performance over stocks database.
# the measurements are taken from tim=3000 to end
# accuracy is messured only on labels != 2.
# generation of labels from prediction matrix:
#                                   label = 1 if (score<=0) else 3;
# correlation is computed as the mean of all cases correlation.
# histograms can be generated using flags. it is saved as a png file
# each histogram is numbered differently
#
# Inputs:
#   prediction 1104X4638 matrix, type of np.array without nan
#   targets 1104X4638X3 matrix, type of np.array
#   plot_hist_flag - if user want to generate histograms
#   plot_hist_path - relative path to where user wants histograms to be saved (must end with '\\'
#######################################################################
def black_box(predictions,targets, train_time_end, plot_hist_flag=False ,plot_hist_path='histograms\\',window1=None):

    #define windows
    if window1 is None:
        if targets.shape[1] == 4779:
            window1 = [4024,4400]
        elif targets.shape[1] == 4638:
            window1 = [3000, 3800]
        else:
            window1 = [train_time_end, train_time_end + (targets.shape[1] - train_time_end)//2]

    if window1[1] != targets.shape[1]:
        window2 = [window1[1], targets.shape[1]]
    else:
        window2 = [window1[1]-1, targets.shape[1]]
    ##### generate matrix according to windows + generate labels matrix #####

    # user predictions
    pred1 = predictions[:,window1[0]:window1[1]]
    pred2 = predictions[:,window2[0]:window2[1]]

    pred_class_1 = np.zeros_like(pred1)
    pred_class_1[np.greater(pred1,0.0)] = 3
    pred_class_1[np.less_equal(pred1,0.0)] = 1

    pred_class_2 = np.zeros_like(pred2)
    pred_class_2[np.greater(pred2, 0.0)] = 3
    pred_class_2[np.less_equal(pred2, 0.0)] = 1

    # real scores and labels
    labels1 = targets[:,window1[0]:window1[1],0]
    labels2 = targets[:,window2[0]:window2[1],0]


    score1 = targets[:,window1[0]:window1[1],2]
    score2 = targets[:,window2[0]:window2[1],2]


    ##### labels accuracya check #####

    # window1
    acc_vec_1 = np.equal(pred_class_1.flatten(),labels1.flatten())
    valid_indices = np.where(np.logical_not(np.logical_or(np.equal(labels1.flatten(),2),np.isnan(labels1.flatten()))))
    acc_vec_1 = acc_vec_1[valid_indices]
    accuracy_window_1 = np.sum(acc_vec_1)/len(acc_vec_1)

    # window2
    acc_vec_2 = np.equal(pred_class_2.flatten(), labels2.flatten())
    valid_indices = np.where(np.logical_not(np.logical_or(np.equal(labels2.flatten(), 2), np.isnan(labels2.flatten()))))
    acc_vec_2 = acc_vec_2[valid_indices]
    accuracy_window_2 = np.sum(acc_vec_2) / len(acc_vec_2)

    # total
    total_accuracy = (np.sum(acc_vec_1)+np.sum(acc_vec_2))/(len(acc_vec_1)+len(acc_vec_2))


    ##### correlation check #####

    num_of_cases = targets.shape[0]
    corr_vec_wind_1 = np.zeros(num_of_cases)
    corr_vec_wind_2 = np.zeros(num_of_cases)
    corr_vec_total = np.zeros(num_of_cases)

    for c in range(num_of_cases):

        # window 1
        case_pred1 = pred1[c,:]
        case_scores1 = score1[c,:]
        corr_vec_wind_1[c] = get_corr(case_pred1,case_scores1)


        # window 2
        case_pred2 = pred2[c,:]
        case_scores2 = score2[c,:]
        corr_vec_wind_2[c] = get_corr(case_pred2,case_scores2)

        # total
        case_pred_tot = np.concatenate((case_pred1,case_pred2),axis=0)
        case_scores_tot = np.concatenate((case_scores1,case_scores2),axis=0)
        corr_vec_total[c] = get_corr(case_pred_tot,case_scores_tot)

    corr_window_1 = np.average(corr_vec_wind_1[~np.isnan(corr_vec_wind_1)])
    corr_window_2 = np.average(corr_vec_wind_2[~np.isnan(corr_vec_wind_2)])
    corr_total = np.average(corr_vec_total[~np.isnan(corr_vec_total)])

    if plot_hist_flag:
        plot_hist(predictions,targets,window1[0],plot_hist_path)

    square_diff = (targets[:,:,2] - predictions) ** 2


    relevant_mat = square_diff[:,:window1[0]]
    relevant_tar = targets[:,:window1[0],2]
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(relevant_tar, 0), np.isnan(relevant_tar))))
    train_rms_loss = np.average(relevant_mat[valid_idx])**0.5

    relevant_mat = square_diff[:, window1[0]:]
    relevant_tar = targets[:,window1[0]:,2]
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(relevant_tar, 0), np.isnan(relevant_tar))))
    test_rms_loss = np.average(relevant_mat[valid_idx])**0.5

    relevant_mat = square_diff[:, window1[0]:window1[1]]
    relevant_tar = targets[:,window1[0]:window1[1],2]
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(relevant_tar, 0), np.isnan(relevant_tar))))
    wind1_rms_loss = np.average(relevant_mat[valid_idx])**0.5

    relevant_mat = square_diff[:, window2[0]:window2[1]]
    relevant_tar = targets[:,window2[0]:window2[1],2]
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(relevant_tar, 0), np.isnan(relevant_tar))))
    wind2_rms_loss = np.average(relevant_mat[valid_idx])**0.5

    return [accuracy_window_1, accuracy_window_2, total_accuracy, corr_window_1, corr_window_2, corr_total,
            train_rms_loss, test_rms_loss, wind1_rms_loss, wind2_rms_loss]




####################################################################
# returns correlation between two vectors
# Inputs:
#       two vectors of the same length
####################################################################
def get_corr(pred_vec,score_vec):
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(score_vec, 0), np.isnan(score_vec))))
    if np.sum(valid_idx) ==0:
        return np.nan

    corr_coeef_mat = np.corrcoef(pred_vec[valid_idx],score_vec[valid_idx])
    return corr_coeef_mat[0,1]


def plot_hist(predictions,targets,test_time_start,path):
    plot_hist.counter+=1

    script_dir = os.path.dirname(__file__)
    hist_dir = os.path.join(script_dir, path)
    if not os.path.isdir(hist_dir):
        os.makedirs(hist_dir)

    if plot_hist.counter==1:
        print('histograms are saved to ' + hist_dir)


    train_score =  targets[:,:test_time_start,2].flatten()
    test_score = targets[:,test_time_start:,2].flatten()

    train_pred = predictions[:,:test_time_start].flatten()
    test_pred = predictions[:,test_time_start:].flatten()

    fig_train = plt.figure()
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(train_score, 0), np.isnan(train_score))))
    n, bins, patches = plt.hist([train_score[valid_idx], train_pred[valid_idx]], bins=100)
    plt.title("train set Histogram")

    file_name = "train_hist_" + str(plot_hist.counter)
    plt.savefig(hist_dir + file_name)

    fig_test = plt.figure()
    valid_idx = np.where(np.logical_not(np.logical_or(np.equal(test_score, 0), np.isnan(test_score))))
    n, bins, patches = plt.hist([test_score[valid_idx], test_pred[valid_idx]], bins=100)
    plt.title("test set Histogram")

    file_name = "test_hist_" + str(plot_hist.counter)
    plt.savefig(hist_dir + file_name)

# static counter
plot_hist.counter = 0
