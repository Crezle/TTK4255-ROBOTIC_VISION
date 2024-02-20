import numpy as np
import matplotlib.pyplot as plt

def plot_all(all_p, all_r, detections, subtract_initial_offset):
    """
    Tip: The logs have been time-synchronized with the image sequence,
    but there may be an offset between the motor angles and the vision
    estimates. You may optionally subtract that offset by passing True
    to subtract_initial_offset.
    """

    #
    # Print reprojection error statistics
    #
    weights = detections[:, ::3]
    reprojection_errors = []
    for i in range(all_p.shape[0]):
        valid = np.reshape(all_r[i], [2,-1])[:, weights[i,:] == 1]
        reprojection_errors.extend(np.linalg.norm(valid, axis=0))
    reprojection_errors = np.array(reprojection_errors)
    print('Reprojection error over whole image sequence:')
    print('- Maximum: %.04f pixels' % np.max(reprojection_errors))
    print('- Average: %.04f pixels' % np.mean(reprojection_errors))
    print('- Median: %.04f pixels' % np.median(reprojection_errors))

    #
    # Figure: Reprojection error distribution
    #
    plt.figure(figsize=(8,3))
    plt.hist(reprojection_errors, bins=80, color='k')
    plt.ylabel('Frequency')
    plt.xlabel('Reprojection error (pixels)')
    plt.title('Reprojection error distribution')
    plt.tight_layout()
    plt.savefig('out_histogram.png')

    #
    # Figure: Comparison between logged encoder values and vision estimates
    #
    logs       = np.loadtxt('../data/logs.txt')
    enc_time   = logs[:,0]
    enc_yaw    = logs[:,1]
    enc_pitch  = logs[:,2]
    enc_roll   = logs[:,3]

    vis_yaw = all_p[:,0]
    vis_pitch = all_p[:,1]
    vis_roll = all_p[:,2]
    if subtract_initial_offset:
        vis_yaw -= vis_yaw[0] - enc_yaw[0]
        vis_pitch -= vis_pitch[0] - enc_pitch[0]
        vis_roll -= vis_roll[0] - enc_roll[0]

    vis_fps  = 16
    enc_frame = enc_time*vis_fps
    vis_frame = np.arange(all_p.shape[0])

    fig,axes = plt.subplots(3, 1, figsize=[6,6], sharex='col')
    axes[0].plot(enc_frame, enc_yaw, 'k:', label='Encoder log')
    axes[0].plot(vis_frame, vis_yaw, 'k', label='Vision estimate')
    axes[0].legend()
    axes[0].set_xlim([0, vis_frame[-1]])
    axes[0].set_ylim([-1, 1])
    axes[0].set_ylabel('Yaw (radians)')

    axes[1].plot(enc_frame, enc_pitch, 'k:')
    axes[1].plot(vis_frame, vis_pitch, 'k')
    axes[1].set_xlim([0, vis_frame[-1]])
    axes[1].set_ylim([0.0, 0.6])
    axes[1].set_ylabel('Pitch (radians)')

    axes[2].plot(enc_frame, enc_roll, 'k:')
    axes[2].plot(vis_frame, vis_roll, 'k')
    axes[2].set_xlim([0, vis_frame[-1]])
    axes[2].set_ylim([-0.6, 0.6])
    axes[2].set_ylabel('Roll (radians)')
    axes[2].set_xlabel('Image number')
    plt.tight_layout()
